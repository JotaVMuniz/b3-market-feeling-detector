"""
Empirical validation of the composite sentiment index (Fear & Greed-style).

Today the weights in src/market_data/compute_composite_index.py (_WEIGHTS) are
set by the author's heuristic judgement, with no statistical backing. This
script adds two independent, data-driven checks against that judgement, using the real
historical data already collected by the pipeline (407 trading days from
2025-01-01 onward):

1. PRINCIPAL COMPONENT ANALYSIS (PCA) of the 6 available raw indicators
   (turnover, TRIN, put/call ratio, % advancing, CDI, consumer confidence —
   CDS Brasil 5y failed to fetch from BCB and is excluded). PCA answers two
   questions: (a) do these 6 indicators actually share a common underlying
   "sentiment" factor at all (variance explained by PC1), and (b) if so, how
   would a data-driven weighting scheme (PC1 loadings) compare to the
   hand-picked heuristic weights currently in production?

2. CONCURRENT / NEXT-DAY VALIDATION against a real market-return benchmark:
   an equal-weighted daily return across the IBrX 100 constituents, computed
   directly from the asset_prices table. This tests whether the composite
   index (heuristic or PCA-based) has any measurable statistical association
   with actual subsequent market performance — the analysis explicitly
   deferred as "future work" until now.

This is exploratory/descriptive statistics on n≈400 daily observations — a
respectable sample for a macro-level daily time series, but still a single
~20-month window (mostly bull-market conditions), so results should be read
as a first empirical pass, not a definitive validation across market cycles.

Usage:
    python scripts/validate_composite_index.py
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_data.compute_composite_index import (
    _WEIGHTS, _INVERTED, _MIN_HISTORY, _WINDOW, _percentile_rank,
)

DB_PATH = "data/news.db"
ALL_INDICATORS = list(_WEIGHTS.keys())


def load_raw_indicators(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, indicator, value FROM sentiment_indicators ORDER BY date", conn
    )
    return df.pivot(index="date", columns="indicator", values="value")


def compute_scores(pivot: pd.DataFrame, indicators) -> pd.DataFrame:
    """Re-derive the same rolling percentile-rank scores used in production."""
    scores = pd.DataFrame(index=pivot.index, columns=indicators, dtype=float)
    history = {k: [] for k in indicators}
    for date in pivot.index:
        for ind in indicators:
            raw = pivot.loc[date, ind] if ind in pivot.columns else None
            hist = history[ind][-_WINDOW:]
            if pd.notna(raw) and len(hist) >= _MIN_HISTORY:
                s = _percentile_rank(raw, hist)
                if ind in _INVERTED:
                    s = 100.0 - s
                scores.loc[date, ind] = s
            if pd.notna(raw):
                history[ind].append(raw)
    return scores


def pca_weights(scores: pd.DataFrame, indicators):
    """
    PCA via eigendecomposition of the correlation matrix.

    Uses pandas' pairwise-complete-observations correlation (the default for
    `.corr()`), NOT a row-wise dropna(): one of the six raw series
    (consumer_confidence, an FGV/ICC monthly survey) is published far less
    often than the other five, which are daily. Requiring all six to be
    non-null on the same calendar day would throw away nearly the entire
    sample. Returns (weights, variance_explained, corr).
    """
    corr = scores.corr()
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    pc1 = eigvecs[:, 0]
    # Orient PC1 so that higher = more "greed" (positive loading on pct_advancing)
    idx_pct_adv = indicators.index("pct_advancing")
    if pc1[idx_pct_adv] < 0:
        pc1 = -pc1

    weights = np.abs(pc1) / np.abs(pc1).sum()
    var_explained = eigvals / eigvals.sum()
    return dict(zip(indicators, weights)), var_explained[0], corr


def market_benchmark_returns(conn) -> pd.Series:
    """Equal-weighted daily return across IBrX 100 constituents."""
    tickers = pd.read_sql_query("SELECT ticker FROM ibrx_tickers", conn)["ticker"].tolist()
    placeholders = ",".join("?" * len(tickers))
    prices = pd.read_sql_query(
        f"SELECT ticker, date, close FROM asset_prices WHERE ticker IN ({placeholders}) AND close IS NOT NULL",
        conn, params=tickers,
    )
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    daily_returns = wide.pct_change()
    return daily_returns.mean(axis=1)  # equal-weighted market proxy


def main():
    conn = sqlite3.connect(DB_PATH)

    pivot = load_raw_indicators(conn)

    available = [ind for ind in ALL_INDICATORS if ind in pivot.columns and pivot[ind].notna().any()]
    missing = [ind for ind in ALL_INDICATORS if ind not in available]
    if missing:
        print(f"Indicadores sem dado algum no período (excluídos do PCA): {missing}\n")

    scores = compute_scores(pivot, available)

    print("=" * 78)
    print("1. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("=" * 78)
    weights_pca, var_pc1, corr = pca_weights(scores, available)
    print("Observações por indicador (algumas séries do Bacen têm cadência mensal, não diária):")
    for ind in available:
        print(f"  {ind:<22}{scores[ind].notna().sum():>5} dias com dado")
    print("(A correlação usada no PCA é pairwise-complete: cada par de indicadores usa "
          "os dias em que ambos têm dado, não a interseção estrita dos 6.)\n")
    print(f"Variância explicada pelo 1º componente principal: {var_pc1 * 100:.1f}%")
    print("(Se alto, os 6 indicadores realmente compartilham um fator comum de "
          "'sentimento'; se baixo, a premissa de um índice único é frágil.)\n")

    print(f"{'Indicador':<22}{'Peso heurístico':>16}{'Peso PCA (PC1)':>16}")
    for ind in available:
        print(f"{ind:<22}{_WEIGHTS[ind]:>16.3f}{weights_pca[ind]:>16.3f}")

    # Heuristic vs PCA composite score correlation
    heur_scores = scores.mul(pd.Series(_WEIGHTS)).sum(axis=1, skipna=True) / scores.notna().mul(pd.Series(_WEIGHTS)).sum(axis=1)
    pca_scores = scores.mul(pd.Series(weights_pca)).sum(axis=1, skipna=True) / scores.notna().mul(pd.Series(weights_pca)).sum(axis=1)
    both = pd.concat([heur_scores.rename("heur"), pca_scores.rename("pca")], axis=1).dropna()
    r_hp, p_hp = stats.pearsonr(both["heur"], both["pca"])
    print(f"\nCorrelação entre índice heurístico (produção) e índice PCA: "
          f"r={r_hp:.3f} (p={p_hp:.1e}), n={len(both)}")

    print()
    print("=" * 78)
    print("2. VALIDAÇÃO CONTRA RETORNO REAL DE MERCADO (benchmark IBrX 100, equal-weighted)")
    print("=" * 78)
    market_ret = market_benchmark_returns(conn)

    combined = pd.concat([
        heur_scores.rename("heur_score"),
        pca_scores.rename("pca_score"),
        market_ret.rename("mkt_ret"),
    ], axis=1).dropna()
    combined["mkt_ret_next"] = combined["mkt_ret"].shift(-1)
    combined["heur_chg"] = combined["heur_score"].diff()

    tests = [
        ("Nível do índice heurístico vs. retorno do MESMO dia", "heur_score", "mkt_ret"),
        ("Nível do índice heurístico vs. retorno do dia SEGUINTE", "heur_score", "mkt_ret_next"),
        ("Nível do índice PCA vs. retorno do MESMO dia", "pca_score", "mkt_ret"),
        ("Nível do índice PCA vs. retorno do dia SEGUINTE", "pca_score", "mkt_ret_next"),
        ("Variação diária do índice heurístico vs. retorno do MESMO dia", "heur_chg", "mkt_ret"),
    ]
    for label, col_a, col_b in tests:
        sub = combined[[col_a, col_b]].dropna()
        if len(sub) < 10:
            print(f"{label}: dados insuficientes (n={len(sub)})")
            continue
        r, p = stats.pearsonr(sub[col_a], sub[col_b])
        sig = "SIGNIFICATIVO (p<0.05)" if p < 0.05 else "não significativo (p>=0.05)"
        print(f"{label}:\n   r={r:+.3f}  p={p:.3f}  n={len(sub)}  -> {sig}")

    print()
    print("Leitura honesta: correlações concorrentes (mesmo dia) tendem a ser "
          "mecânicas, já que turnover/TRIN/PCR/% em alta são derivados dos mesmos "
          "preços usados no benchmark. O teste que importa de verdade é o de "
          "PODER PREDITIVO (índice hoje x retorno de amanhã) — reportado acima "
          "sem maquiagem, seja ele significativo ou não.")


if __name__ == "__main__":
    main()
