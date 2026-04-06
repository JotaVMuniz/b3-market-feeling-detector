"""
Compute price-movement metrics for news items that contain ticker references.

For each (news, ticker) pair the following variations are calculated:

- ``d0_var``: intraday variation on the news publication date
  ``(close - open) / open``.
- ``d1_var``: next-trading-day close vs. previous-day close
  ``(d1_close - d0_close) / d0_close``.
- ``d5_var``: 5-trading-days-later close vs. news-day close
  ``(d5_close - d0_close) / d0_close``.

Dates that fall on weekends or holidays are mapped to the nearest available
trading day found in the ``asset_prices`` table.
"""

import datetime
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Z]{4}\d{1,2}$")


def _is_valid_ticker(ticker: str) -> bool:
    """Return True if *ticker* matches the B3 format (4 letters + 1-2 digits)."""
    return bool(_TICKER_RE.match(ticker))


def _parse_tickers(tickers_json: Optional[str]) -> List[str]:
    """Parse JSON-encoded tickers field from the news table."""
    if not tickers_json:
        return []
    try:
        tickers = json.loads(tickers_json)
        if isinstance(tickers, list):
            return [t for t in tickers if isinstance(t, str) and _is_valid_ticker(t)]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _find_nearest_date(
    target_date: str,
    available_dates: List[str],
    direction: str = "forward",
) -> Optional[str]:
    """
    Find the closest available trading date to *target_date*.

    Args:
        target_date: ISO date string to look up.
        available_dates: Sorted list of ISO date strings that have price data.
        direction: ``"forward"`` searches on or after the target date;
            ``"backward"`` searches on or before.

    Returns:
        The nearest available ISO date string, or None.
    """
    if not available_dates:
        return None

    target = datetime.date.fromisoformat(target_date)
    sorted_dates = sorted(available_dates)

    if direction == "forward":
        for d in sorted_dates:
            if datetime.date.fromisoformat(d) >= target:
                return d
    else:
        for d in reversed(sorted_dates):
            if datetime.date.fromisoformat(d) <= target:
                return d
    return None


def _nth_trading_date_after(
    base_date: str,
    n: int,
    available_dates: List[str],
) -> Optional[str]:
    """
    Return the *n*-th trading date strictly after *base_date*.

    Args:
        base_date: ISO date string of the reference date (D0).
        n: Number of trading sessions to advance.
        available_dates: Sorted list of ISO date strings with price data.

    Returns:
        ISO date string or None if not enough future dates exist.
    """
    sorted_dates = sorted(available_dates)
    base = datetime.date.fromisoformat(base_date)
    future = [d for d in sorted_dates if datetime.date.fromisoformat(d) > base]
    if len(future) >= n:
        return future[n - 1]
    return None


def _price_variation(open_price: Optional[float], close_price: Optional[float]) -> Optional[float]:
    """Return ``(close - open) / open`` or None if either value is missing/zero."""
    if open_price is None or close_price is None:
        return None
    if open_price == 0:
        return None
    return (close_price - open_price) / open_price


def _close_to_close_variation(
    prev_close: Optional[float],
    curr_close: Optional[float],
) -> Optional[float]:
    """Return ``(curr_close - prev_close) / prev_close`` or None."""
    if prev_close is None or curr_close is None:
        return None
    if prev_close == 0:
        return None
    return (curr_close - prev_close) / prev_close


def compute_correlations(
    news_records: List[Dict],
    market_db,
) -> List[Dict]:
    """
    Compute D0/D+1/D+5 price-sentiment correlations for a batch of news.

    Only news that have:
    - a valid ``published_at`` date,
    - at least one recognised ticker,
    - a non-null ``sentiment`` value,

    …are processed.

    Args:
        news_records: Rows from the ``news`` table (dicts with at least
            ``id``, ``published_at``, ``tickers``, ``sentiment``,
            ``confidence``).
        market_db: A :class:`MarketDatabase` instance used to look up prices.

    Returns:
        List of correlation dicts ready to be passed to
        :meth:`MarketDatabase.upsert_correlations`.
    """
    results: List[Dict] = []

    for news in news_records:
        tickers = _parse_tickers(news.get("tickers"))
        if not tickers:
            continue

        sentiment = news.get("sentiment")
        if not sentiment:
            continue

        raw_date = news.get("published_at", "")
        if not raw_date:
            continue

        try:
            news_date = datetime.date.fromisoformat(raw_date[:10])
        except ValueError:
            logger.debug(f"Could not parse date '{raw_date}' for news id={news.get('id')}")
            continue

        for ticker in tickers:
            correlation = _compute_single_correlation(
                news_id=news["id"],
                ticker=ticker,
                news_date=news_date,
                sentiment=sentiment,
                confidence=news.get("confidence"),
                market_db=market_db,
            )
            if correlation is not None:
                results.append(correlation)

    logger.info(f"Computed {len(results)} correlation records")
    return results


def _compute_single_correlation(
    news_id: int,
    ticker: str,
    news_date: datetime.date,
    sentiment: str,
    confidence: Optional[float],
    market_db,
) -> Optional[Dict]:
    """
    Compute the correlation record for a single (news, ticker) pair.

    Returns None when not enough price data is available.
    """
    # Collect available dates for this ticker from the DB
    date_from = (news_date - datetime.timedelta(days=1)).isoformat()
    date_to = (news_date + datetime.timedelta(days=14)).isoformat()

    price_rows = market_db.get_prices_for_ticker(ticker, date_from, date_to)
    if not price_rows:
        return None

    available_dates = [row["date"] for row in price_rows]
    price_map: Dict[str, Dict] = {row["date"]: row for row in price_rows}

    # D0: trading day on or after the news publication date
    d0_date = _find_nearest_date(news_date.isoformat(), available_dates, direction="forward")
    if d0_date is None:
        return None

    d0_row = price_map.get(d0_date)
    d0_var = _price_variation(
        d0_row.get("open") if d0_row else None,
        d0_row.get("close") if d0_row else None,
    )

    # D+1: next trading day after D0
    d1_date = _nth_trading_date_after(d0_date, 1, available_dates)
    d1_row = price_map.get(d1_date) if d1_date else None
    d1_var = _close_to_close_variation(
        d0_row.get("close") if d0_row else None,
        d1_row.get("close") if d1_row else None,
    )

    # D+5: fifth trading day after D0
    d5_date = _nth_trading_date_after(d0_date, 5, available_dates)
    d5_row = price_map.get(d5_date) if d5_date else None
    d5_var = _close_to_close_variation(
        d0_row.get("close") if d0_row else None,
        d5_row.get("close") if d5_row else None,
    )

    return {
        "news_id": news_id,
        "ticker": ticker,
        "news_date": news_date.isoformat(),
        "sentiment": sentiment,
        "confidence": confidence,
        "d0_var": d0_var,
        "d1_var": d1_var,
        "d5_var": d5_var,
    }
