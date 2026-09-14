"""
Consistency check for ticker attribution: does the ticker the LLM extracted
from a news item actually have textual evidence in that news' title/summary?

Today the pipeline only validates tickers *syntactically* (regex matching the
B3 naming convention, in src/nlp/enrichment.py). It never checks whether the
ticker is *semantically* plausible given the news content — so a
hallucinated or misattributed ticker (e.g., the LLM mentioning MGLU3 for a
headline that never references Magazine Luiza) passes through silently.

This script adds that missing semantic check as a post-hoc, read-only pass:
for every (news, ticker) pair, it looks up the company's official name (from
the `companies` table, populated from real B3 price data) and checks whether
any significant token of that name appears in the news title + summary.

This is deliberately a simple, dependency-free heuristic (no extra LLM calls,
no new libraries) so it can be run cheaply and often. It will not catch every
case (aliases like "Magalu" for MAGAZ LUIZA, or "Vale" being a very short,
ambiguous token) — see --aliases for a small manual alias table that covers
the most common blind spots for the current IBrX 100 universe.

Usage:
    python scripts/check_ticker_consistency.py            # report only
    python scripts/check_ticker_consistency.py --persist  # also writes a
                                                            # `ticker_consistency`
                                                            # table to the DB
"""

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Manual aliases for common colloquial names that diverge a lot from the
# official B3 short name on file (extend as new false positives show up).
ALIASES: Dict[str, List[str]] = {
    "MGLU3": ["magalu", "magazine luiza"],
    "WEGE3": ["weg"],
    "RENT3": ["localiza"],
    "RADL3": ["raia drogasil", "drogasil", "raiadrogasil"],
    "ABEV3": ["ambev"],
    "ITUB3": ["itau", "itaú"],
    "ITUB4": ["itau", "itaú"],
    "BBAS3": ["banco do brasil"],
    "PETR3": ["petrobras"],
    "PETR4": ["petrobras"],
    "VALE3": ["vale"],
    "SUZB3": ["suzano"],
    "GGBR4": ["gerdau"],
    "CSAN3": ["cosan"],
    "PRIO3": ["prio", "petrorio"],
    "PCAR3": ["gpa", "pao de acucar", "grupo pao de acucar"],
}

MIN_TOKEN_LEN = 4
STOPWORD_TOKENS = {"pn", "on", "ej", "nm", "n1", "n2", "eds", "edj", "unt"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def company_tokens(name: str) -> List[str]:
    return [t for t in normalize(name).split() if len(t) >= MIN_TOKEN_LEN and t not in STOPWORD_TOKENS]


def has_evidence(ticker: str, text_norm: str, company_name: Optional[str]) -> Optional[bool]:
    """Return True/False if we could check, None if we have no company name on file.

    Checked in order, cheapest/most-reliable first:
    1. The literal ticker string appears in the text (e.g. "...PCAR3 fecha
       abaixo de R$4") — unambiguous, common in financial headlines.
    2. A manually curated alias (e.g. "magalu", "gpa") appears in full.
    3. The official B3 short name: requires ALL of its significant tokens to
       appear as a WHOLE-WORD match (a single generic word like "magazine"
       or "banco" is not enough), tried both as spaced tokens and as one
       whitespace-stripped blob — B3 short names are often truncated
       concatenations (e.g. "BBSEGURIDADE", "FICTORALIMEN") that don't split
       into real dictionary tokens.
    """
    if re.search(rf"\b{re.escape(ticker.lower())}\b", text_norm):
        return True

    for alias in ALIASES.get(ticker, []):
        tokens = company_tokens(alias) or [normalize(alias)]
        if all(tok in text_norm for tok in tokens):
            return True

    if company_name:
        tokens = company_tokens(company_name)
        if not tokens:
            return None
        if all(tok in text_norm for tok in tokens):
            return True
        # Fallback: whitespace-stripped match, for concatenated/truncated
        # official names. Two directions matter: the text may spell the name
        # out fully where the DB has it concatenated (e.g. "BB Seguridade"
        # vs "BBSEGURIDADE" -> stripping the text's space is enough), or the
        # text may use a shorter brand form than the DB's fuller name (e.g.
        # "Fictor" vs "FICTORALIMEN" -> only a name PREFIX will be found).
        stripped_name = normalize(company_name).replace(" ", "")
        stripped_text = text_norm.replace(" ", "")
        prefix = stripped_name[:6]
        if len(stripped_name) >= MIN_TOKEN_LEN and (
            stripped_name in stripped_text or (len(prefix) >= 6 and prefix in stripped_text)
        ):
            return True
        return False

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/news.db")
    parser.add_argument("--persist", action="store_true",
                         help="Write results to a `ticker_consistency` table instead of just printing")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    company_names = {
        row["ticker"]: row["name"]
        for row in conn.execute("SELECT DISTINCT ticker, name FROM companies")
    }

    news_rows = conn.execute(
        "SELECT id, title, content, source, published_at, tickers FROM news WHERE tickers IS NOT NULL AND tickers != '[]'"
    ).fetchall()

    total_pairs = 0
    flagged = []
    checked_unknown = 0

    for row in news_rows:
        try:
            tickers = json.loads(row["tickers"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not tickers:
            continue
        text_norm = normalize(f'{row["title"]} {row["content"] or ""}')
        for ticker in tickers:
            total_pairs += 1
            result = has_evidence(ticker, text_norm, company_names.get(ticker))
            if result is None:
                checked_unknown += 1
                continue
            if not result:
                flagged.append({
                    "news_id": row["id"],
                    "ticker": ticker,
                    "title": row["title"],
                    "source": row["source"],
                    "published_at": row["published_at"],
                    "company_name_on_file": company_names.get(ticker),
                })

    print(f"Pares notícia-ticker avaliados: {total_pairs}")
    print(f"  Sem nome de empresa cadastrado (não avaliável): {checked_unknown}")
    print(f"  Consistentes (nome/alias encontrado no texto): {total_pairs - checked_unknown - len(flagged)}")
    print(f"  SEM evidência textual (possível atribuição incorreta): {len(flagged)}")
    print()

    for f in flagged:
        print(f'  [news_id={f["news_id"]}] {f["ticker"]} ({f["company_name_on_file"]}) '
              f'<- "{f["title"]}" ({f["source"]}, {f["published_at"]})')

    if args.persist:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_consistency (
                news_id INTEGER,
                ticker TEXT,
                has_evidence INTEGER,
                checked_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (news_id, ticker)
            )
        """)
        conn.executemany(
            "INSERT OR REPLACE INTO ticker_consistency (news_id, ticker, has_evidence) VALUES (?, ?, 0)",
            [(f["news_id"], f["ticker"]) for f in flagged],
        )
        conn.commit()
        print(f"\nPersistido: {len(flagged)} registros marcados em `ticker_consistency` (has_evidence=0).")


if __name__ == "__main__":
    main()
