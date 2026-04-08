"""
Fetch fundamental financial indicators for B3 assets.

Data sources
------------
- **yfinance**: P/L, P/VPA, EV/EBITDA, ROE, Margem Líquida, ROA,
  Dívida/PL, Liquidez Corrente, Dividend Yield, Payout.
  Brazilian tickers require the ``.SA`` suffix on Yahoo Finance
  (e.g. ``PETR4`` → ``PETR4.SA``).

- **mercados.bcb**: Selic meta (taxa alvo anual) and IPCA 12-month
  variation.  These macroeconomic indicators are stored under the
  special ticker ``"__MACRO__"`` so the dashboard can display them
  as a market-context panel applicable to every asset.

- **mercados.b3**: FII dividend history for real estate investment
  funds.  Used to compute the 12-month Dividend Yield for FIIs
  when the yfinance value is missing or zero.
"""

import datetime
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yfinance field → (internal_key, portuguese_label) mapping
# ---------------------------------------------------------------------------

_YF_FIELD_MAP: Dict[str, tuple] = {
    "trailingPE":         ("pl",                "P/L"),
    "priceToBook":        ("pvpa",              "P/VPA"),
    "enterpriseToEbitda": ("ev_ebitda",         "EV/EBITDA"),
    "returnOnEquity":     ("roe",               "ROE"),
    "profitMargins":      ("margem_liquida",    "Margem Líquida"),
    "returnOnAssets":     ("roa",               "ROA"),
    "debtToEquity":       ("divida_pl",         "Dívida/PL"),
    "currentRatio":       ("liquidez_corrente", "Liquidez Corrente"),
    "dividendYield":      ("dy",                "Dividend Yield"),
    "payoutRatio":        ("payout",            "Payout"),
}


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def fetch_asset_fundamentals(ticker: str) -> List[Dict]:
    """
    Fetch fundamental indicators for a single B3 ticker via yfinance.

    The ticker is automatically suffixed with ``.SA`` when querying
    Yahoo Finance (e.g. ``PETR4`` → ``PETR4.SA``).

    Args:
        ticker: B3 ticker code without the ``.SA`` suffix.

    Returns:
        List of dicts with keys ``ticker``, ``key``, ``value``,
        ``label``, ``updated_at``.  Returns an empty list when the
        ticker is not found on Yahoo Finance or when yfinance returns
        no usable data.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — run: pip install yfinance")
        return []

    yf_symbol = f"{ticker}.SA"
    today = datetime.date.today().isoformat()

    try:
        info = yf.Ticker(yf_symbol).info
    except Exception as exc:
        logger.warning("yfinance error for %s: %s", yf_symbol, exc)
        return []

    # An empty or minimal response means the ticker was not found
    if not info or info.get("quoteType") is None:
        logger.debug("No yfinance data for %s", yf_symbol)
        return []

    results: List[Dict] = []
    for yf_key, (internal_key, label) in _YF_FIELD_MAP.items():
        raw = info.get(yf_key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (ValueError, TypeError):
            continue
        results.append({
            "ticker":     ticker,
            "key":        internal_key,
            "value":      value,
            "label":      label,
            "updated_at": today,
        })

    logger.info(
        "Fetched %d fundamental indicators for %s via yfinance",
        len(results),
        ticker,
    )
    return results


def fetch_fundamentals_for_tickers(tickers: List[str]) -> List[Dict]:
    """
    Fetch fundamental indicators for a list of B3 tickers.

    Args:
        tickers: List of B3 ticker codes (without ``.SA``).

    Returns:
        Flat list of indicator dicts, one per (ticker, indicator) pair.
    """
    results: List[Dict] = []
    for ticker in tickers:
        results.extend(fetch_asset_fundamentals(ticker))
    logger.info(
        "fetch_fundamentals_for_tickers: %d records for %d tickers",
        len(results),
        len(tickers),
    )
    return results


# ---------------------------------------------------------------------------
# mercados.bcb macro indicators
# ---------------------------------------------------------------------------

def fetch_macro_fundamentals() -> List[Dict]:
    """
    Fetch current Selic target rate and IPCA 12-month variation from BCB.

    These macro indicators are stored under the special ticker
    ``"__MACRO__"`` so the dashboard can display them as market context
    for all assets without being tied to a specific ticker.

    Data is sourced from the BCB SGS time-series API via
    ``mercados.bcb.BancoCentral``:

    - **Selic meta diária** (series 432): annual target rate set by COPOM.
    - **IPCA mensal** (series 433): monthly index value; 12-month variation
      is computed as ``(latest / value_12_months_ago - 1) * 100``.

    Returns:
        List of indicator dicts with ``ticker = "__MACRO__"``.
    """
    try:
        from mercados.bcb import BancoCentral
    except ImportError:
        logger.error("mercados.bcb not available — skipping macro fundamentals")
        return []

    today = datetime.date.today()
    results: List[Dict] = []
    bc = BancoCentral()

    # ------------------------------------------------------------------
    # Selic meta rate (annual %)
    # ------------------------------------------------------------------
    try:
        start = today - datetime.timedelta(days=60)
        rates = list(bc.serie_temporal("Selic meta diária", inicio=start, fim=today))
        if rates:
            latest = rates[-1]
            results.append({
                "ticker":     "__MACRO__",
                "key":        "selic_meta",
                "value":      float(latest.valor),
                "label":      "Selic Meta (% a.a.)",
                "updated_at": latest.data.isoformat(),
            })
    except Exception as exc:
        logger.warning("Could not fetch Selic meta rate: %s", exc)

    # ------------------------------------------------------------------
    # IPCA 12-month variation
    # ------------------------------------------------------------------
    try:
        start_ipca = today - datetime.timedelta(days=400)
        ipca = list(bc.serie_temporal("IPCA mensal", inicio=start_ipca, fim=today))
        if len(ipca) >= 2:
            # series 433 returns monthly % changes (e.g. 0.39 for 0.39%)
            # sum the last 12 months to get the 12-month accumulated rate
            last_12 = ipca[-12:] if len(ipca) >= 12 else ipca
            # compound the monthly rates
            accumulated = 1.0
            for entry in last_12:
                accumulated *= (1 + float(entry.valor) / 100)
            variation_12m = round((accumulated - 1) * 100, 2)
            results.append({
                "ticker":     "__MACRO__",
                "key":        "ipca_12m",
                "value":      variation_12m,
                "label":      "IPCA 12 meses (%)",
                "updated_at": ipca[-1].data.isoformat(),
            })
    except Exception as exc:
        logger.warning("Could not fetch IPCA: %s", exc)

    logger.info("Fetched %d macro fundamental records via BCB", len(results))
    return results


# ---------------------------------------------------------------------------
# mercados.b3 FII dividend yield supplement
# ---------------------------------------------------------------------------

def _is_fii_ticker(ticker: str) -> bool:
    """Return True when *ticker* looks like a FII (ends with '11')."""
    return ticker.upper().endswith("11")


def fetch_fii_dy_supplement(
    ticker: str,
    current_price: Optional[float],
) -> Optional[Dict]:
    """
    Compute the 12-month Dividend Yield for a FII using ``mercados.b3``.

    This is used as a fallback / cross-check when yfinance does not
    provide a ``dividendYield`` value for the FII.

    Args:
        ticker:        B3 ticker code (e.g. ``"XPML11"``).
        current_price: Latest closing price in BRL.  When ``None`` the
                       DY cannot be computed and ``None`` is returned.

    Returns:
        A single indicator dict, or ``None`` when data is unavailable.
    """
    if not _is_fii_ticker(ticker) or not current_price or current_price <= 0:
        return None

    try:
        from mercados.b3 import B3
    except ImportError:
        return None

    today = datetime.date.today()
    one_year_ago = today - datetime.timedelta(days=365)

    try:
        b3 = B3()
        detail = b3.fii_detail(ticker)
        cnpj = detail.cnpj if detail else None
        if not cnpj:
            return None

        dividends = b3.fii_dividends(cnpj, ticker)
        annual_income = sum(
            float(d.valor_por_cota)
            for d in dividends
            if d.data_pagamento and d.data_pagamento >= one_year_ago
        )
        if annual_income <= 0:
            return None

        dy = round(annual_income / current_price * 100, 2)
        logger.info(
            "FII DY supplement for %s: annual_income=%.4f price=%.2f DY=%.2f%%",
            ticker,
            annual_income,
            current_price,
            dy,
        )
        return {
            "ticker":     ticker,
            "key":        "dy",
            "value":      dy / 100,  # store as decimal to match yfinance convention
            "label":      "Dividend Yield",
            "updated_at": today.isoformat(),
        }
    except Exception as exc:
        logger.warning("Could not fetch FII dividend data for %s: %s", ticker, exc)
        return None
