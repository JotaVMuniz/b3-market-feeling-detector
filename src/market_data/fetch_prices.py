"""
Fetch daily stock prices from B3 using the mercados library.
"""

import datetime
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def fetch_daily_prices(date: datetime.date) -> List[Dict]:
    """
    Fetch all daily traded prices from B3 for a given date.

    Args:
        date: Trading date to fetch.

    Returns:
        List of price records as dicts with keys:
        ticker, date, open, close, high, low, avg_price, volume.
        Returns an empty list when there is no trading session for that date,
        when the date is in the future, or when the request fails.
    """
    today = datetime.date.today()
    if date > today:
        logger.debug(f"Skipping future date {date} (no B3 data available yet)")
        return []

    try:
        from mercados.b3 import B3

        b3 = B3()
        result = b3.negociacao_bolsa("dia", date)

        if isinstance(result, ValueError):
            logger.warning(f"No trading data available for {date}: {result}")
            return []

        prices = []
        for neg in result:
            prices.append({
                "ticker": neg.codigo_negociacao,
                "date": date.isoformat(),
                "open": float(neg.preco_abertura) if neg.preco_abertura is not None else None,
                "close": float(neg.preco_ultimo) if neg.preco_ultimo is not None else None,
                "high": float(neg.preco_maximo) if neg.preco_maximo is not None else None,
                "low": float(neg.preco_minimo) if neg.preco_minimo is not None else None,
                "avg_price": float(neg.preco_medio) if neg.preco_medio is not None else None,
                "volume": float(neg.volume) if neg.volume is not None else None,
                "nome_pregao": neg.nome_pregao,
                "tipo_papel": neg.tipo_papel,
                "codigo_isin": neg.codigo_isin,
            })

        logger.info(f"Fetched {len(prices)} price records for {date}")
        return prices

    except Exception as e:
        logger.error(f"Error fetching prices for {date}: {str(e)}")
        return []


def fetch_prices_for_tickers(
    tickers: List[str],
    dates: List[datetime.date],
) -> List[Dict]:
    """
    Fetch prices for specific tickers on a list of dates.

    Only dates up to and including today are fetched; future dates are silently
    skipped because B3 historical files are not yet available.

    Args:
        tickers: B3 ticker codes to filter.
        dates: Dates for which prices are needed.

    Returns:
        List of price records for the requested tickers and dates.
    """
    if not tickers or not dates:
        return []

    today = datetime.date.today()
    ticker_set = set(tickers)
    results = []

    for date in sorted(set(dates)):
        if date > today:
            logger.debug(f"Skipping future date {date} in fetch_prices_for_tickers")
            continue
        daily_prices = fetch_daily_prices(date)
        for price in daily_prices:
            if price["ticker"] in ticker_set:
                results.append(price)

    return results


def next_trading_day(
    start: datetime.date,
    max_lookahead: int = 10,
) -> Optional[datetime.date]:
    """
    Return the first trading day on or after *start* by trying to fetch
    prices and returning the first date that yields data.

    Args:
        start: Starting date.
        max_lookahead: Maximum number of calendar days to look ahead.

    Returns:
        A date with trading activity, or None if none found within the window.
    """
    for offset in range(max_lookahead + 1):
        candidate = start + datetime.timedelta(days=offset)
        # Quick weekend skip to reduce unnecessary requests
        if candidate.weekday() >= 5:
            continue
        prices = fetch_daily_prices(candidate)
        if prices:
            return candidate
    return None
