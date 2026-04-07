"""
Fetch sentiment indicators from the mercados library.

Indicators computed from B3 daily trading data (``mercados.b3``):

- **turnover**: Total financial volume traded on the market for the day
  (sum of ``volume`` across all stocks). A proxy for market liquidity and
  investor activity.
- **trin** (Arms Index / TRIN): ``(N_advancing / N_declining) /
  (V_advancing / V_declining)``. Values above 1 indicate more volume
  concentrated in declining stocks (bearish signal).
- **put_call_ratio** (PCR): ``volume_puts / volume_calls`` computed from
  listed options (``codigo_tipo_mercado == 70`` for calls,
  ``codigo_tipo_mercado == 80`` for puts). Values above 1 indicate
  pessimism.
- **pct_advancing**: Fraction of stocks whose closing price is above their
  opening price on the day.

Indicator computed from BCB time-series data (``mercados.bcb``):

- **cdi_rate**: Daily CDI rate from the BCB SGS API, used as a proxy for
  domestic short-term interest-rate pressure (related to DI amplitude).

These raw values are stored in the ``sentiment_indicators`` table and later
normalised by :mod:`src.market_data.compute_composite_index` to produce the
composite sentiment index.
"""

import datetime
import logging
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# B3 indicators
# ---------------------------------------------------------------------------


def fetch_market_indicators(date: datetime.date) -> Optional[Dict]:
    """
    Compute market-sentiment indicators for *date* from B3 daily prices.

    Returns ``None`` when no trading data is available for that date (holiday,
    weekend, future date) or when the data cannot be fetched.

    Args:
        date: Trading date to analyse.

    Returns:
        Dict with keys ``date``, ``turnover``, ``trin``,
        ``put_call_ratio``, ``pct_advancing``; or ``None``.
    """
    today = datetime.date.today()
    if date > today:
        logger.debug("Skipping future date %s for market indicators", date)
        return None

    try:
        from mercados.b3 import B3

        b3 = B3()
        records = list(b3.negociacao_bolsa("dia", date))
    except Exception as exc:
        logger.error("Error fetching B3 data for %s: %s", date, exc)
        return None

    if not records:
        logger.debug("No B3 trading data for %s", date)
        return None

    # ------------------------------------------------------------------
    # 1. Turnover — total financial volume of the session (stocks only)
    # ------------------------------------------------------------------
    # Use codigo_tipo_mercado == 10 (Vista) to restrict to regular stocks
    stock_records = [
        r for r in records
        if r.codigo_tipo_mercado is not None and r.codigo_tipo_mercado == 10
    ]
    if not stock_records:
        # Fall back to all records if no "Vista" stocks are present
        stock_records = records

    total_volume = sum(
        float(r.volume) for r in stock_records if r.volume is not None
    )

    # ------------------------------------------------------------------
    # 2. TRIN (Arms Index)
    # ------------------------------------------------------------------
    advancing = [
        r for r in stock_records
        if r.preco_ultimo is not None
        and r.preco_abertura is not None
        and r.preco_ultimo > r.preco_abertura
    ]
    declining = [
        r for r in stock_records
        if r.preco_ultimo is not None
        and r.preco_abertura is not None
        and r.preco_ultimo < r.preco_abertura
    ]

    n_adv = len(advancing)
    n_dec = len(declining)
    v_adv = sum(float(r.volume) for r in advancing if r.volume is not None)
    v_dec = sum(float(r.volume) for r in declining if r.volume is not None)

    trin: Optional[float] = None
    if n_dec > 0 and v_dec > 0 and v_adv > 0:
        trin = (n_adv / n_dec) / (v_adv / v_dec)

    # ------------------------------------------------------------------
    # 3. Put-Call Ratio (PCR)
    # ------------------------------------------------------------------
    # codigo_tipo_mercado: 70 = Opções de Compra (calls), 80 = Opções de Venda (puts)
    call_records = [
        r for r in records
        if r.codigo_tipo_mercado is not None and r.codigo_tipo_mercado == 70
    ]
    put_records = [
        r for r in records
        if r.codigo_tipo_mercado is not None and r.codigo_tipo_mercado == 80
    ]
    v_calls = sum(float(r.volume) for r in call_records if r.volume is not None)
    v_puts = sum(float(r.volume) for r in put_records if r.volume is not None)

    put_call_ratio: Optional[float] = None
    if v_calls > 0:
        put_call_ratio = v_puts / v_calls

    # ------------------------------------------------------------------
    # 4. Percentage of advancing stocks
    # ------------------------------------------------------------------
    stocks_with_prices = [
        r for r in stock_records
        if r.preco_ultimo is not None and r.preco_abertura is not None
    ]
    pct_advancing: Optional[float] = None
    if stocks_with_prices:
        n_advancing = sum(
            1 for r in stocks_with_prices if r.preco_ultimo > r.preco_abertura
        )
        pct_advancing = n_advancing / len(stocks_with_prices)

    logger.info(
        "Market indicators for %s: turnover=%.2e, trin=%s, pcr=%s, pct_adv=%s",
        date,
        total_volume,
        f"{trin:.4f}" if trin is not None else "N/A",
        f"{put_call_ratio:.4f}" if put_call_ratio is not None else "N/A",
        f"{pct_advancing:.4f}" if pct_advancing is not None else "N/A",
    )

    return {
        "date": date.isoformat(),
        "turnover": total_volume,
        "trin": trin,
        "put_call_ratio": put_call_ratio,
        "pct_advancing": pct_advancing,
    }


def fetch_market_indicators_range(
    start_date: datetime.date,
    end_date: Optional[datetime.date] = None,
) -> List[Dict]:
    """
    Fetch market sentiment indicators for every trading day in a date range.

    Args:
        start_date: First date to fetch (inclusive).
        end_date:   Last date to fetch (inclusive). Defaults to today.

    Returns:
        List of indicator dicts (one per trading day with available data).
    """
    today = datetime.date.today()
    if end_date is None:
        end_date = today
    end_date = min(end_date, today)

    if start_date > end_date:
        return []

    results = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday–Friday
            indicators = fetch_market_indicators(current)
            if indicators is not None:
                results.append(indicators)
        current += datetime.timedelta(days=1)

    logger.info(
        "fetch_market_indicators_range(%s → %s): %d records",
        start_date,
        end_date,
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# BCB indicators
# ---------------------------------------------------------------------------

# SGS series codes used as additional macroeconomic indicators
_BCB_SERIES: Dict[str, int] = {
    # CDI – already in mercados.bcb but we fetch it here as well
    "CDI": 12,
    # ICC-FGV (Índice de Confiança do Consumidor – FGV/Fecomercio)
    "ICC": 4393,
    # CDS Brasil 5Y (USD) – proxy for country risk
    "CDS_5Y": 28229,
}


def fetch_bcb_indicators(
    date_from: datetime.date,
    date_to: datetime.date,
) -> List[Dict]:
    """
    Fetch BCB time-series indicators between *date_from* and *date_to*.

    Each returned record has ``date`` (ISO string), ``indicator`` (series name)
    and ``value`` (float).

    Currently fetches:
    - ``cdi_rate`` – daily CDI rate (proxy for DI amplitude)
    - ``consumer_confidence`` – ICC consumer confidence index (monthly)
    - ``cds_brasil_5y`` – CDS Brazil 5Y spread (when available)

    Args:
        date_from: Start of the period (inclusive).
        date_to:   End of the period (inclusive).

    Returns:
        List of indicator dicts ready for :meth:`~MarketDatabase.upsert_indicators`.
    """
    try:
        from mercados.bcb import BancoCentral
    except ImportError:
        logger.error("mercados.bcb not available — skipping BCB indicators")
        return []

    bc = BancoCentral()
    results: List[Dict] = []

    for series_name, indicator_key in [
        ("CDI", "cdi_rate"),
        ("ICC", "consumer_confidence"),
        ("CDS_5Y", "cds_brasil_5y"),
    ]:
        series_code = _BCB_SERIES[series_name]
        # Allow adding new SGS codes not in the default mercados series dict
        if series_name not in bc.series:
            bc.series[series_name] = series_code

        try:
            for taxa in bc.serie_temporal(series_name, inicio=date_from, fim=date_to):
                results.append({
                    "date": taxa.data.isoformat(),
                    "indicator": indicator_key,
                    "value": float(taxa.valor),
                })
        except Exception as exc:
            logger.warning(
                "Could not fetch BCB series '%s' (code %d): %s",
                series_name,
                series_code,
                exc,
            )

    logger.info(
        "fetch_bcb_indicators(%s → %s): %d records",
        date_from,
        date_to,
        len(results),
    )
    return results
