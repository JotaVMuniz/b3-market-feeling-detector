"""
Fetch and build company/asset metadata from B3 via the mercados library.

The mercados library does not expose a dedicated "listed stocks" endpoint,
so company metadata is derived from the daily price files (NegociacaoBolsa
records contain nome_pregao, tipo_papel, and codigo_isin).
"""

import datetime
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def extract_companies_from_prices(price_records: List[Dict]) -> List[Dict]:
    """
    Derive company metadata from a list of price records returned by
    :func:`src.market_data.fetch_prices.fetch_daily_prices`.

    Each unique ticker yields one company record.  When a ticker appears
    multiple times only the first occurrence is kept.

    Args:
        price_records: Price dicts that include ``ticker``, ``nome_pregao``,
            ``tipo_papel``, and ``codigo_isin``.

    Returns:
        List of company dicts with keys: ticker, name, tipo_papel, isin.
    """
    seen: set = set()
    companies: List[Dict] = []

    for record in price_records:
        ticker = record.get("ticker", "").strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        companies.append({
            "ticker": ticker,
            "name": record.get("nome_pregao", "").strip(),
            "tipo_papel": record.get("tipo_papel", "").strip(),
            "isin": record.get("codigo_isin", "").strip(),
        })

    logger.info(f"Extracted {len(companies)} unique company records from price data")
    return companies


def fetch_companies_from_recent_trading(days_back: int = 5) -> List[Dict]:
    """
    Build a company listing by fetching recent trading sessions.

    Tries up to *days_back* working days backwards from yesterday to find a
    session with data, then returns all unique companies found in that session.

    Args:
        days_back: Number of previous calendar days to scan.

    Returns:
        List of company metadata dicts.
    """
    from src.market_data.fetch_prices import fetch_daily_prices

    today = datetime.date.today()

    for offset in range(1, days_back + 8):
        candidate = today - datetime.timedelta(days=offset)
        if candidate.weekday() >= 5:  # skip weekends
            continue
        prices = fetch_daily_prices(candidate)
        if prices:
            logger.info(f"Using trading session {candidate} to build company list")
            return extract_companies_from_prices(prices)

    logger.warning("Could not find a recent trading session to build company list")
    return []
