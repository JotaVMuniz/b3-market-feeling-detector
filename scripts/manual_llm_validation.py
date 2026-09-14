"""
Manual semantic validation of the LLM enrichment output.

Automated checks in this pipeline validate *structure* (regex-valid ticker
codes, sentiment restricted to a controlled vocabulary) and, in
scripts/check_ticker_consistency.py, a heuristic textual-evidence match for
tickers. None of that is a substitute for a human independently judging
whether the model's semantic reading of a news item — sentiment, financial
relevance, and ticker attribution — is actually correct. This script
implements that missing manual validation step.

Methodology:
  1. A fixed-seed random sample of already-enriched news is drawn once and
     persisted to a CSV, so the sample is reproducible and stable across
     resumed sessions.
  2. For sentiment and financial relevance, the reviewer judges each item
     BLIND — the LLM's answer is not shown until after the human has
     committed to a judgment — so the agreement rate approximates a genuine
     inter-rater reliability measure, not a biased confirmation check.
  3. For tickers, blind independent recall is unrealistic (a human will not
     reliably recall exact B3 ticker codes from memory), so the LLM's
     proposed tickers are shown and the reviewer confirms/rejects each one
     against the article text — a precision-style check, consistent with
     the automated check_ticker_consistency.py methodology, but based on
     genuine human reading rather than string matching.

Usage:
    python scripts/manual_llm_validation.py                  # label interactively
    python scripts/manual_llm_validation.py --sample-size 40  # first run only
    python scripts/manual_llm_validation.py --report          # print agreement stats
    python scripts/manual_llm_validation.py --report --csv path/to/other.csv

Progress is saved after every single item, so the session can be safely
interrupted (Ctrl+C or `q`) and resumed later without losing labels.
"""

import argparse
import csv
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

CSV_FIELDS = [
    "news_id", "title", "content_preview", "source", "published_at",
    "llm_sentiment", "llm_confidence", "llm_is_relevant", "llm_market_relevance",
    "llm_tickers",
    "human_sentiment", "human_is_relevant", "human_tickers_confirmed", "human_tickers_rejected",
    "agree_sentiment", "agree_relevance", "notes",
]

SENTIMENT_CHOICES = {"positivo", "negativo", "neutro"}


def load_sample_from_db(db_path: str, sample_size: int, seed: int) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, content, source, published_at, sentiment, confidence, "
        "is_relevant, market_relevance, tickers "
        "FROM news WHERE sentiment IS NOT NULL"
    ).fetchall()
    conn.close()

    if not rows:
        print("Nenhuma notícia enriquecida encontrada no banco (sentiment IS NOT NULL).")
        sys.exit(1)

    rng = random.Random(seed)
    sampled = rng.sample(list(rows), k=min(sample_size, len(rows)))

    out = []
    for r in sampled:
        try:
            tickers = json.loads(r["tickers"]) if r["tickers"] else []
        except (json.JSONDecodeError, TypeError):
            tickers = []
        content = (r["content"] or "").strip()
        out.append({
            "news_id": r["id"],
            "title": r["title"],
            "content_preview": content[:1200],
            "source": r["source"],
            "published_at": r["published_at"],
            "llm_sentiment": r["sentiment"],
            "llm_confidence": r["confidence"],
            "llm_is_relevant": r["is_relevant"],
            "llm_market_relevance": r["market_relevance"],
            "llm_tickers": json.dumps(tickers, ensure_ascii=False),
            "human_sentiment": "",
            "human_is_relevant": "",
            "human_tickers_confirmed": "",
            "human_tickers_rejected": "",
            "agree_sentiment": "",
            "agree_relevance": "",
            "notes": "",
        })
    return out


def load_csv(path: Path) -> List[Dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: List[Dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    tmp.replace(path)


def prompt_choice(question: str, choices: set) -> str:
    while True:
        ans = input(question).strip().lower()
        if ans == "q":
            return "q"
        if ans in choices:
            return ans
        print(f"  Resposta inválida. Opções: {', '.join(sorted(choices))} (ou 'q' para salvar e sair)")


def run_interactive(rows: List[Dict], csv_path: Path) -> None:
    pending = [r for r in rows if not r["human_sentiment"]]
    total = len(rows)
    done = total - len(pending)

    if not pending:
        print(f"Amostra completa: {total}/{total} notícias já rotuladas. "
              f"Rode com --report para ver as taxas de concordância.")
        return

    print(f"\nValidação manual — {done}/{total} já rotuladas, {len(pending)} restantes.")
    print("Digite 'q' a qualquer momento para salvar o progresso e sair.\n")

    for row in pending:
        print("=" * 78)
        print(f'[news_id={row["news_id"]}] {row["source"]} — {row["published_at"]}')
        print(f'Título: {row["title"]}')
        if row["content_preview"]:
            print(f'Conteúdo: {row["content_preview"][:600]}')
        print("-" * 78)

        # --- Blind judgment (no LLM answer shown yet) ------------------
        sentiment = prompt_choice(
            "Na sua leitura, o sentimento desta notícia é (positivo/negativo/neutro): ",
            SENTIMENT_CHOICES,
        )
        if sentiment == "q":
            break

        relevant = prompt_choice(
            "Esta notícia é financeiramente relevante para o mercado? (s/n): ",
            {"s", "n"},
        )
        if relevant == "q":
            break

        # --- Reveal LLM tickers, ask for confirmation (precision check) --
        try:
            llm_tickers = json.loads(row["llm_tickers"]) if row["llm_tickers"] else []
        except (json.JSONDecodeError, TypeError):
            llm_tickers = []

        confirmed, rejected = [], []
        if llm_tickers:
            print(f"\nO modelo atribuiu os tickers: {', '.join(llm_tickers)}")
            for t in llm_tickers:
                ans = prompt_choice(f"  '{t}' está corretamente associado a esta notícia? (s/n): ", {"s", "n"})
                if ans == "q":
                    save_csv(csv_path, rows)
                    print(f"\nProgresso salvo em {csv_path}.")
                    return
                (confirmed if ans == "s" else rejected).append(t)
        else:
            print("\nO modelo não atribuiu nenhum ticker a esta notícia.")

        # --- Reveal LLM's own sentiment/relevance answer, compute agreement --
        llm_sentiment = (row["llm_sentiment"] or "").strip().lower()
        llm_relevant = "s" if str(row["llm_is_relevant"]) in ("1", "True", "true") else "n"

        agree_sentiment = "sim" if sentiment == llm_sentiment else "não"
        agree_relevance = "sim" if relevant == llm_relevant else "não"

        print(f"\nResposta do modelo — sentimento: {llm_sentiment or 'n/d'} "
              f"(confiança {row['llm_confidence']}); relevante: {llm_relevant}")
        print(f"Concordância — sentimento: {agree_sentiment}; relevância: {agree_relevance}")

        notes = input("Observações (opcional, Enter para pular): ").strip()

        row["human_sentiment"] = sentiment
        row["human_is_relevant"] = relevant
        row["human_tickers_confirmed"] = ";".join(confirmed)
        row["human_tickers_rejected"] = ";".join(rejected)
        row["agree_sentiment"] = agree_sentiment
        row["agree_relevance"] = agree_relevance
        row["notes"] = notes

        save_csv(csv_path, rows)

    remaining = len([r for r in rows if not r["human_sentiment"]])
    labeled = len(rows) - remaining
    print(f"\nProgresso salvo em {csv_path}: {labeled}/{len(rows)} rotuladas "
          f"({remaining} restantes).")


def print_report(rows: List[Dict]) -> None:
    labeled = [r for r in rows if r["human_sentiment"]]
    total = len(rows)
    n = len(labeled)
    print(f"Amostra: {n}/{total} notícias rotuladas manualmente.\n")
    if not n:
        print("Nenhuma notícia rotulada ainda. Rode sem --report para começar a validação.")
        return

    agree_sent = sum(1 for r in labeled if r["agree_sentiment"] == "sim")
    agree_rel = sum(1 for r in labeled if r["agree_relevance"] == "sim")

    total_tickers = confirmed_tickers = rejected_tickers = 0
    for r in labeled:
        confirmed = [t for t in (r["human_tickers_confirmed"] or "").split(";") if t]
        rejected = [t for t in (r["human_tickers_rejected"] or "").split(";") if t]
        confirmed_tickers += len(confirmed)
        rejected_tickers += len(rejected)
        total_tickers += len(confirmed) + len(rejected)

    print(f"Concordância de sentimento (humano vs. LLM, julgamento cego): "
          f"{agree_sent}/{n} ({100 * agree_sent / n:.1f}%)")
    print(f"Concordância de relevância financeira (humano vs. LLM, julgamento cego): "
          f"{agree_rel}/{n} ({100 * agree_rel / n:.1f}%)")
    if total_tickers:
        precision = 100 * confirmed_tickers / total_tickers
        print(f"Precisão da atribuição de tickers (confirmados pelo humano / total avaliado): "
              f"{confirmed_tickers}/{total_tickers} ({precision:.1f}%)")
    else:
        print("Nenhum ticker avaliado na amostra rotulada até agora.")

    disagreements = [r for r in labeled if r["agree_sentiment"] == "não" or r["agree_relevance"] == "não"]
    if disagreements:
        print(f"\n{len(disagreements)} caso(s) de discordância humano-LLM "
              f"(sentimento e/ou relevância) — vale revisar manualmente:")
        for r in disagreements:
            print(f'  [news_id={r["news_id"]}] humano={r["human_sentiment"]}/{r["human_is_relevant"]} '
                  f'vs. LLM={r["llm_sentiment"]}/{"s" if str(r["llm_is_relevant"]) in ("1","True","true") else "n"} '
                  f'— "{r["title"][:80]}"')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="data/news.db")
    parser.add_argument("--csv", default="data/manual_llm_validation_sample.csv",
                         help="Arquivo onde a amostra e os rótulos são persistidos")
    parser.add_argument("--sample-size", type=int, default=30,
                         help="Tamanho da amostra (usado apenas na primeira execução)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed do sorteio aleatório (para reprodutibilidade)")
    parser.add_argument("--report", action="store_true",
                         help="Apenas imprime as taxas de concordância da amostra já rotulada")
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if csv_path.exists():
        rows = load_csv(csv_path)
    else:
        if args.report:
            print(f"Arquivo {csv_path} não existe ainda — rode sem --report para sortear a amostra.")
            sys.exit(1)
        rows = load_sample_from_db(args.db, args.sample_size, args.seed)
        save_csv(csv_path, rows)
        print(f"Amostra de {len(rows)} notícias sorteada (seed={args.seed}) e salva em {csv_path}.")

    if args.report:
        print_report(rows)
    else:
        run_interactive(rows, csv_path)


if __name__ == "__main__":
    main()
