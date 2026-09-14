"""
Report que junta as três tabelas analíticas do pipeline:

- news_price_correlation (variação de preço D0/D1/D5 por par notícia-ativo)
- composite_sentiment_index (índice composto de sentimento do dia da reação, D0)
- asset_fundamentals (indicadores fundamentalistas do ticker)

Uso:
    python scripts/news_price_report.py [--db data/news.db] [--min-conf 0.0]

Este script é apenas leitura: não altera o banco de dados. Ele é pensado para
ser reexecutado periodicamente conforme a tabela news_price_correlation cresce
com as execuções diárias do pipeline (main.py --stage all).
"""

import argparse
import sqlite3
from typing import Optional

import numpy as np
from scipy import stats


FUND_HIGHLIGHTS = ["pl", "pvpa", "roe", "dy", "ev_ebitda"]


def get_d0_index(conn: sqlite3.Connection, news_date: str) -> Optional[sqlite3.Row]:
    cur = conn.execute(
        "SELECT date, score, label FROM composite_sentiment_index "
        "WHERE date >= ? ORDER BY date LIMIT 1",
        (news_date,),
    )
    return cur.fetchone()


def get_fundamentals(conn: sqlite3.Connection, ticker: str) -> dict:
    cur = conn.execute(
        "SELECT key, value, label FROM asset_fundamentals WHERE ticker = ? AND key IN (%s)"
        % ",".join("?" * len(FUND_HIGHLIGHTS)),
        (ticker, *FUND_HIGHLIGHTS),
    )
    return {row["key"]: (row["value"], row["label"]) for row in cur.fetchall()}


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/d"
    return f"{value * 100:+.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/news.db", help="Caminho do banco SQLite")
    parser.add_argument("--min-conf", type=float, default=0.0,
                         help="Confiança mínima do sentimento para incluir no relatório")
    parser.add_argument("--exclude-flagged", action="store_true",
                         help="Exclui pares notícia-ticker sinalizados por scripts/check_ticker_consistency.py "
                              "--persist (atribuições de ticker sem evidência textual)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    def make_exclude_clause(alias: str) -> str:
        if not args.exclude_flagged:
            return ""
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ticker_consistency'"
        ).fetchone()
        if not has_table:
            return ""
        return f"""
            AND NOT EXISTS (
                SELECT 1 FROM ticker_consistency tc
                WHERE tc.news_id = {alias}.news_id AND tc.ticker = {alias}.ticker AND tc.has_evidence = 0
            )
        """

    if args.exclude_flagged and not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ticker_consistency'"
    ).fetchone():
        print("Aviso: --exclude-flagged pedido, mas a tabela ticker_consistency não existe ainda "
              "(rode `python scripts/check_ticker_consistency.py --persist` primeiro). Ignorando filtro.\n")

    exclude_clause = make_exclude_clause("c")

    rows = conn.execute(
        f"""
        SELECT c.ticker, c.news_date, c.sentiment, c.confidence,
               c.d0_var, c.d1_var, c.d5_var, n.title, n.source
        FROM news_price_correlation c
        JOIN news n ON n.id = c.news_id
        WHERE c.confidence >= ?
        {exclude_clause}
        ORDER BY c.news_date DESC, c.ticker
        """,
        (args.min_conf,),
    ).fetchall()

    print(f"Total de pares notícia-ativo com correlação calculada: {len(rows)}\n")
    if not rows:
        print("Nenhum registro ainda. Rode `python main.py --stage all` por alguns dias "
              "para acumular pares notícia-ativo (RSS só traz notícias do próprio dia).")
        return

    for r in rows:
        idx = get_d0_index(conn, r["news_date"])
        fund = get_fundamentals(conn, r["ticker"])

        print("=" * 78)
        print(f'{r["ticker"]} — "{r["title"]}" ({r["source"]}, {r["news_date"]})')
        print(f'  Sentimento da notícia: {r["sentiment"]} (confiança {r["confidence"]:.2f})')
        print(f'  Variação D0:  {fmt_pct(r["d0_var"])}')
        print(f'  Variação D+1: {fmt_pct(r["d1_var"])}')
        print(f'  Variação D+5: {fmt_pct(r["d5_var"])}')
        if idx:
            print(f'  Índice composto de sentimento no D0 ({idx["date"]}): '
                  f'{idx["score"]:.1f} ({idx["label"]})')
        else:
            print("  Índice composto de sentimento no D0: n/d")
        if fund:
            fund_str = ", ".join(
                f'{label}={value:.2f}' for key, (value, label) in fund.items()
            )
            print(f"  Fundamentos ({r['ticker']}): {fund_str}")
        else:
            print(f"  Fundamentos ({r['ticker']}): n/d")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Agregado por sentimento (informativo apenas — médias brutas podem ser
    # distorcidas por outliers; o teste de hipótese abaixo, baseado em
    # medianas, é o que de fato avalia significância estatística)
    # ------------------------------------------------------------------
    print("\nMédia de variação por sentimento (apenas descritivo, sem teste estatístico):")
    agg_exclude = make_exclude_clause("news_price_correlation")
    agg = conn.execute(
        f"""
        SELECT sentiment, COUNT(*) as n,
               AVG(d0_var) as avg_d0, AVG(d1_var) as avg_d1, AVG(d5_var) as avg_d5
        FROM news_price_correlation
        WHERE confidence >= ?
        {agg_exclude}
        GROUP BY sentiment
        """,
        (args.min_conf,),
    ).fetchall()
    for a in agg:
        print(f'  {a["sentiment"]:>10} (n={a["n"]:>3}): '
              f'D0={fmt_pct(a["avg_d0"])}  D+1={fmt_pct(a["avg_d1"])}  D+5={fmt_pct(a["avg_d5"])}')

    # ------------------------------------------------------------------
    # Teste estatístico real: positivo vs. negativo, por horizonte.
    # Mann-Whitney U (não paramétrico) em vez de t-test, pois variações de
    # preço tendem a ter caudas pesadas / não-normalidade, mesmo com n>20.
    # ------------------------------------------------------------------
    print("\nTeste de hipótese (Mann-Whitney U): notícias positivas vs. negativas têm "
          "distribuição de variação diferente?")
    pos_rows = conn.execute(
        f"SELECT d0_var, d1_var, d5_var FROM news_price_correlation c "
        f"WHERE sentiment='positivo' AND confidence >= ? {exclude_clause}", (args.min_conf,)
    ).fetchall()
    neg_rows = conn.execute(
        f"SELECT d0_var, d1_var, d5_var FROM news_price_correlation c "
        f"WHERE sentiment='negativo' AND confidence >= ? {exclude_clause}", (args.min_conf,)
    ).fetchall()

    for col, label in [("d0_var", "D0"), ("d1_var", "D+1"), ("d5_var", "D+5")]:
        pos = [r[col] for r in pos_rows if r[col] is not None]
        neg = [r[col] for r in neg_rows if r[col] is not None]
        if len(pos) < 5 or len(neg) < 5:
            print(f"  {label}: amostra insuficiente (positivo n={len(pos)}, negativo n={len(neg)})")
            continue
        u_stat, p_value = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        sig = "SIGNIFICATIVO (p<0.05)" if p_value < 0.05 else "não significativo (p>=0.05)"
        print(f"  {label}: positivo n={len(pos)} (mediana {fmt_pct(float(np.median(pos)))}), "
              f"negativo n={len(neg)} (mediana {fmt_pct(float(np.median(neg)))}) "
              f"-> U={u_stat:.1f}, p={p_value:.3f} -> {sig}")

    print("\nInterpretação: com n ainda modesto (dezenas, não centenas, por grupo), um "
          "resultado não significativo não prova ausência de efeito — só que não temos "
          "poder estatístico suficiente ainda. O ideal é reexecutar este teste conforme "
          "a coleta diária (main.py --stage all) for acumulando mais pares notícia-ativo.")


if __name__ == "__main__":
    main()
