"""
Fetch fundamental financial indicators for B3 assets.

Data sources
------------
- **Fundamentus** (primary): P/L, P/VP, EV/EBITDA, EV/EBIT, ROE, ROA, ROIC,
  Margem Líquida, Margem Bruta, Margem EBIT, Dívida/PL, Liquidez Corrente,
  Dividend Yield, Cotação, P/EBIT, and more.
  Scraped from ``https://www.fundamentus.com.br/detalhes.php?papel=<TICKER>``.

- **yfinance** (fallback / supplement): Fields not available in Fundamentus
  (e.g. Payout).  Brazilian tickers require the ``.SA`` suffix on Yahoo Finance
  (e.g. ``PETR4`` → ``PETR4.SA``).

- **mercados.bcb**: Selic meta (taxa alvo anual) and IPCA 12-month
  variation.  These macroeconomic indicators are stored under the
  special ticker ``"__MACRO__"`` so the dashboard can display them
  as a market-context panel applicable to every asset.

- **mercados.b3**: FII dividend history for real estate investment
  funds.  Used to compute the 12-month Dividend Yield for FIIs
  when neither Fundamentus nor yfinance provides a usable value.
"""

import datetime
import logging
import re
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
# Fundamentus scraping
# ---------------------------------------------------------------------------

_FUNDAMENTUS_BASE_URL = "https://www.fundamentus.com.br/detalhes.php"

# Mapping from Fundamentus page label → (internal_key, portuguese_label, is_pct)
# is_pct=True means the value is expressed as a percentage on the page and
# must be divided by 100 to match the yfinance decimal convention.
_FUNDAMENTUS_FIELD_MAP: Dict[str, tuple] = {
    "P/L":                ("pl",                "P/L",               False),
    "P/VP":               ("pvpa",              "P/VPA",             False),
    "P/EBIT":             ("p_ebit",            "P/EBIT",            False),
    "EV/EBIT":            ("ev_ebit",           "EV/EBIT",           False),
    "EV/EBITDA":          ("ev_ebitda",         "EV/EBITDA",         False),
    "PSR":                ("psr",               "PSR",               False),
    "P/Ativo":            ("p_ativo",           "P/Ativo",           False),
    "P/Cap. Giro":        ("p_cap_giro",        "P/Cap. Giro",       False),
    "P/Ativ Circ Liq":    ("p_ativ_circ_liq",  "P/Ativ Circ Liq",   False),
    "Div. Yield":         ("dy",                "Dividend Yield",    True),
    "ROE":                ("roe",               "ROE",               True),
    "ROIC":               ("roic",              "ROIC",              True),
    "ROA":                ("roa",               "ROA",               True),
    "Giro Ativos":        ("giro_ativos",       "Giro Ativos",       False),
    "Marg. Bruta":        ("margem_bruta",      "Margem Bruta",      True),
    "Marg. EBIT":         ("margem_ebit",       "Margem EBIT",       True),
    "Marg. Líquida":      ("margem_liquida",    "Margem Líquida",    True),
    "Liq. Corr.":         ("liquidez_corrente", "Liquidez Corrente", False),
    "Dív. Bruta/Patrim.": ("divida_pl",         "Dívida/PL",         False),
    "Cresc. Rec. 5a.":    ("cresc_receita_5a",  "Cresc. Rec. 5a.",   True),
    "Cotação":            ("preco",             "Cotação",           False),
}

_FUNDAMENTUS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.fundamentus.com.br/",
}


def _parse_br_number(text: str) -> Optional[float]:
    """
    Parse a Brazilian-formatted number string into a Python float.

    Handles formats like ``"1.234,56"``, ``"12,34%"``, ``"-0,45%"``,
    ``"(1.234,56)"``.  Returns ``None`` when the value cannot be parsed
    (e.g. ``"-"`` or empty string).

    Args:
        text: Raw text extracted from a Fundamentus HTML cell.

    Returns:
        Float value or ``None``.
    """
    if not text:
        return None
    t = text.strip()
    if not t or t == "-":
        return None
    # Remove % sign and leading/trailing whitespace
    t = t.replace("%", "").strip()
    # Handle parentheses notation for negative numbers: (1.234,56) → -1234.56
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    # Brazilian format: dot = thousands separator, comma = decimal
    t = t.replace(".", "").replace(",", ".")
    # Remove any leftover non-numeric characters except leading minus and dot
    t = re.sub(r"[^\d\.\-]", "", t)
    try:
        return float(t)
    except ValueError:
        return None


def fetch_fundamentus_data(ticker: str) -> List[Dict]:
    """
    Scrape fundamental indicators for a single B3 ticker from Fundamentus.

    Fetches ``https://www.fundamentus.com.br/detalhes.php?papel=<TICKER>``
    and extracts all recognised label/value pairs from the HTML tables.

    **Parsing strategy** (in order of preference):

    1. *Class-based*: finds every ``<td class="label">`` cell and pairs it
       with the immediately following ``<td class="data">`` sibling.  This
       matches the Fundamentus DOM exactly.
    2. *Alternating-cell fallback*: walks every table row and treats
       adjacent cell pairs as ``(label, value)``.  Used when the class-based
       strategy yields nothing (e.g. in unit tests with minimal HTML).

    **Encoding**: Fundamentus always serves pages in ISO-8859-1.  Using
    ``resp.content`` (raw bytes) with ``from_encoding="iso-8859-1"``
    bypasses ``chardet`` / ``apparent_encoding`` misdetection that would
    garble accented characters such as those in ``"Marg. Líquida"`` and
    ``"Cotação"``, preventing them from matching ``_FUNDAMENTUS_FIELD_MAP``.

    Args:
        ticker: B3 ticker code without suffix (e.g. ``"PETR4"``).

    Returns:
        List of indicator dicts (same schema as
        :func:`fetch_asset_fundamentals`).  Returns an empty list when the
        ticker is not found on Fundamentus or when the request fails.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        logger.error("Missing dependency for Fundamentus scraping: %s", exc)
        return []

    url = f"{_FUNDAMENTUS_BASE_URL}?papel={ticker}"
    try:
        resp = requests.get(url, headers=_FUNDAMENTUS_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Fundamentus request failed for %s: %s", ticker, exc)
        return []

    try:
        # Pass raw bytes with the known encoding so BeautifulSoup/chardet
        # cannot misdetect it and garble accented Portuguese labels.
        soup = BeautifulSoup(resp.content, "html.parser", from_encoding="iso-8859-1")
    except Exception as exc:
        logger.warning("Failed to parse Fundamentus HTML for %s: %s", ticker, exc)
        return []

    raw_data: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Strategy 1: class-based (matches real Fundamentus DOM structure).
    # Labels are in <td class="label"> cells; values in the immediately
    # following <td class="data"> sibling within the same <tr>.
    # ------------------------------------------------------------------
    for label_td in soup.find_all("td", class_="label"):
        label_text = label_td.get_text(strip=True).rstrip("?").strip()
        data_td = label_td.find_next_sibling("td", class_="data")
        if data_td and label_text:
            raw_data[label_text] = data_td.get_text(strip=True)

    # ------------------------------------------------------------------
    # Strategy 2: alternating-cell fallback (used when no class-based
    # cells were found, e.g. in unit tests with plain <td> markup).
    # ------------------------------------------------------------------
    if not raw_data:
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                for i in range(0, len(cells) - 1, 2):
                    label = cells[i].get_text(strip=True).rstrip("?").strip()
                    value = cells[i + 1].get_text(strip=True)
                    if label and value:
                        raw_data[label] = value

    logger.debug(
        "Fundamentus raw labels for %s (%d found): %s",
        ticker,
        len(raw_data),
        list(raw_data.keys()),
    )

    if not raw_data:
        logger.warning("No data extracted from Fundamentus for %s", ticker)
        return []

    today = datetime.date.today().isoformat()
    results: List[Dict] = []
    for fund_label, (internal_key, label, is_pct) in _FUNDAMENTUS_FIELD_MAP.items():
        raw = raw_data.get(fund_label)
        if raw is None:
            continue
        value = _parse_br_number(raw)
        if value is None:
            continue
        if is_pct:
            value = round(value / 100.0, 6)
        results.append({
            "ticker":     ticker,
            "key":        internal_key,
            "value":      value,
            "label":      label,
            "updated_at": today,
        })

    logger.info(
        "Fetched %d fundamental indicators for %s via Fundamentus",
        len(results),
        ticker,
    )
    return results


# ---------------------------------------------------------------------------
# Primary fetch: Fundamentus + yfinance supplement
# ---------------------------------------------------------------------------

def fetch_asset_fundamentals(ticker: str) -> List[Dict]:
    """
    Fetch fundamental indicators for a single B3 ticker.

    **Fundamentus** is queried first as the primary source.  Any keys not
    returned by Fundamentus are then supplemented using **yfinance** (e.g.
    Payout ratio, which Fundamentus does not publish).

    The ticker is automatically suffixed with ``.SA`` when querying Yahoo
    Finance (e.g. ``PETR4`` → ``PETR4.SA``).

    Args:
        ticker: B3 ticker code without the ``.SA`` suffix.

    Returns:
        List of dicts with keys ``ticker``, ``key``, ``value``,
        ``label``, ``updated_at``.  Returns an empty list only when both
        Fundamentus and yfinance return no usable data.
    """
    # --- Step 1: Fundamentus (primary) ----------------------------------------
    results = fetch_fundamentus_data(ticker)
    covered_keys = {r["key"] for r in results}

    # --- Step 2: yfinance supplement for missing keys -------------------------
    missing_yf_keys = {
        yf_key: (internal_key, label)
        for yf_key, (internal_key, label) in _YF_FIELD_MAP.items()
        if internal_key not in covered_keys
    }

    if not missing_yf_keys:
        return results

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — run: pip install yfinance")
        return results

    yf_symbol = f"{ticker}.SA"
    today = datetime.date.today().isoformat()

    try:
        info = yf.Ticker(yf_symbol).info
    except Exception as exc:
        logger.warning("yfinance error for %s: %s", yf_symbol, exc)
        return results

    if not info or info.get("quoteType") is None:
        logger.debug("No yfinance data for %s", yf_symbol)
        return results

    for yf_key, (internal_key, label) in missing_yf_keys.items():
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
        "Fetched %d fundamental indicators for %s (Fundamentus + yfinance)",
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

        dy = round(annual_income / current_price, 4)
        logger.info(
            "FII DY supplement for %s: annual_income=%.4f price=%.2f DY=%.2f%%",
            ticker,
            annual_income,
            current_price,
            dy * 100,
        )
        return {
            "ticker":     ticker,
            "key":        "dy",
            "value":      dy,  # stored as decimal to match yfinance convention
            "label":      "Dividend Yield",
            "updated_at": today.isoformat(),
        }
    except Exception as exc:
        logger.warning("Could not fetch FII dividend data for %s: %s", ticker, exc)
        return None
