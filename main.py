"""
Main orchestration module for the financial news ingestion pipeline.

The pipeline is split into five independent stages that can be run separately:

- raw:       Fetch news from RSS feeds, clean and persist raw records.
- trusted:   Enrich stored news with NLP/LLM sentiment analysis.
- prices:    Fetch B3 market prices (fully independent of news data).
- analytics: Compute news–price correlations from stored data.
- backfill:  Download full B3 price history for a date range (all assets).

Usage::

    python main.py --stage raw
    python main.py --stage trusted
    python main.py --stage trusted --reprocess-existing
    python main.py --stage prices
    python main.py --stage prices --tickers PETR4,VALE3 --date 2026-04-01
    python main.py --stage analytics
    python main.py --stage backfill
    python main.py --stage backfill --from 2025-01-01 --to 2025-12-31
    python main.py --stage all          # full pipeline (default)
    python main.py --stage all --no-market-data
"""

import argparse
import datetime
import json as _json
import logging
import sys
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

# Import pipeline modules
from src.ingestion.sources import get_sources
from src.ingestion.fetch_news import fetch_all_news, deduplicate_news
from src.processing.clean_news import clean_news_batch, validate_news_entry
from src.storage.save_raw import save_raw_news
from src.storage.database import NewsDatabase
from src.nlp.enrichment import enrich_batch
from src.market_data.database_market import MarketDatabase
from src.market_data.fetch_prices import fetch_prices_for_tickers, fetch_all_prices_range
from src.market_data.fetch_companies import extract_companies_from_prices
from src.market_data.correlation import compute_correlations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_tickers_from_news(news_records: List[dict]) -> set:
    """Return the set of unique tickers mentioned across *news_records*."""
    tickers: set = set()
    for news in news_records:
        raw = news.get("tickers")
        if isinstance(raw, list):
            tickers.update(t for t in raw if t)
        elif isinstance(raw, str):
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, list):
                    tickers.update(t for t in parsed if t)
            except Exception:
                pass
    return tickers


def _tickers_nonempty(val) -> bool:
    """Return True when *val* represents a non-empty tickers list."""
    if not val:
        return False
    if isinstance(val, list):
        return bool(val)
    try:
        return bool(_json.loads(val))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def run_raw() -> None:
    """
    Stage 1 – Raw: fetch news from RSS feeds, clean and persist.

    Produces:
    - A JSON snapshot under ``data/raw/news_YYYYMMDD.json``
    - Basic (unenriched) news records inserted into the ``news`` table
    """
    logger.info("=" * 60)
    logger.info("Stage: RAW — fetching and storing news")
    logger.info("=" * 60)

    sources = get_sources()
    logger.info(f"Configured sources: {len(sources)}")

    raw_news = fetch_all_news(sources)
    logger.info(f"Total news fetched: {len(raw_news)}")

    if not raw_news:
        logger.warning("No news fetched. Stage RAW complete.")
        return

    deduplicated = deduplicate_news(raw_news)
    logger.info(f"After deduplication: {len(deduplicated)}")

    cleaned = clean_news_batch(deduplicated)
    valid_news = [n for n in cleaned if validate_news_entry(n)]
    logger.info(f"Valid entries: {len(valid_news)}")

    if not valid_news:
        logger.warning("No valid news entries after cleaning. Stage RAW complete.")
        return

    raw_file = save_raw_news(valid_news)
    logger.info(f"Raw data saved to: {raw_file}")

    db = NewsDatabase()
    inserted = db.insert_news(valid_news)
    logger.info(f"Inserted {inserted} records into news table")

    logger.info("Stage RAW complete\n")


def run_trusted(reprocess_all: bool = False) -> None:
    """
    Stage 2 – Trusted: enrich stored news with NLP/LLM sentiment analysis.

    By default only records without enrichment (``sentiment IS NULL``) are
    processed.  Pass ``reprocess_all=True`` to re-enrich every record.

    Args:
        reprocess_all: When True re-enrich all existing records in the DB.
    """
    logger.info("=" * 60)
    logger.info("Stage: TRUSTED — enriching news with NLP/LLM")
    logger.info("=" * 60)

    db = NewsDatabase()

    if reprocess_all:
        to_enrich = db.get_all_news()
        logger.info(f"Reprocessing all {len(to_enrich)} news records")
    else:
        to_enrich = db.get_news_without_enrichment()
        logger.info(f"Found {len(to_enrich)} unenriched news records")

    if not to_enrich:
        logger.info("No news to enrich. Stage TRUSTED complete.")
        return

    texts = [
        f"{row.get('title', '')} {row.get('content', '')}".strip()
        for row in to_enrich
    ]

    enrichment_results = enrich_batch(texts)

    updated_rows = [
        {
            "url": row.get("url", ""),
            "is_relevant": enrichment["is_relevant"],
            "sentiment": enrichment["sentiment"],
            "confidence": enrichment["confidence"],
            "segments": enrichment["segments"],
            "tickers": enrichment["tickers"],
        }
        for row, enrichment in zip(to_enrich, enrichment_results)
    ]

    updated = db.update_news_batch(updated_rows)
    relevant = sum(1 for e in enrichment_results if e["is_relevant"])
    logger.info(f"Updated {updated} records — {relevant} marked as relevant")

    logger.info("Stage TRUSTED complete\n")


def run_prices(
    tickers: Optional[List[str]] = None,
    dates: Optional[List[datetime.date]] = None,
) -> None:
    """
    Stage 3 – Prices: fetch B3 market prices (independent of news data).

    This stage runs independently from news enrichment.  When *tickers* is
    ``None`` every ticker found in the ``news`` and ``companies`` tables is
    used.  When *dates* is ``None`` the publication dates of stored news
    (plus a D+1 through D+5 lookahead) are used, capped at today.

    Args:
        tickers: Explicit list of B3 ticker codes.  Auto-detected if ``None``.
        dates:   Explicit list of trading dates.  Auto-detected if ``None``.
    """
    logger.info("=" * 60)
    logger.info("Stage: PRICES — fetching B3 market data")
    logger.info("=" * 60)

    db = NewsDatabase()
    market_db = MarketDatabase(db_path=db.db_path)

    # Fetch all news once if auto-detection is needed for either argument
    _all_news: Optional[List[dict]] = None
    if tickers is None or dates is None:
        _all_news = db.get_all_news()

    if tickers is None:
        tickers_set = _collect_tickers_from_news(_all_news)
        tickers_set.update(market_db.get_known_tickers())
        tickers = list(tickers_set)

    if not tickers:
        logger.warning("No tickers found — skipping PRICES stage")
        return

    logger.info(f"Tickers to fetch ({len(tickers)}): {tickers}")

    if dates is None:
        today = datetime.date.today()
        dates_set: set = set()
        for news in _all_news:
            raw_date = news.get("published_at", "")
            if raw_date:
                try:
                    dates_set.add(datetime.date.fromisoformat(raw_date[:10]))
                except ValueError:
                    pass

        if not dates_set:
            # Fallback: last 7 calendar days
            for offset in range(7):
                dates_set.add(today - datetime.timedelta(days=offset))

        # Extend with D+1..D+5 lookahead for correlation calculation
        extended: set = set(dates_set)
        for base in list(dates_set):
            for offset in range(1, 8):
                extended.add(base + datetime.timedelta(days=offset))

        dates = sorted(d for d in extended if d <= today)

    if not dates:
        logger.warning("No valid dates to fetch — skipping PRICES stage")
        return

    logger.info(f"Date range: {dates[0]} → {dates[-1]} ({len(dates)} dates)")

    price_records = fetch_prices_for_tickers(tickers=tickers, dates=dates)

    if price_records:
        prices_written = market_db.upsert_prices(price_records)
        logger.info(f"Stored {prices_written} price records in asset_prices")

        companies = extract_companies_from_prices(price_records)
        companies_written = market_db.upsert_companies(companies)
        logger.info(f"Stored {companies_written} company records in companies")
    else:
        logger.warning("No price data returned for the requested tickers/dates")

    logger.info("Stage PRICES complete\n")


def run_backfill(
    start_date: datetime.date,
    end_date: Optional[datetime.date] = None,
) -> None:
    """
    Backfill stage – fetch all B3 prices for every trading day in a date range.

    Downloads the complete daily price file from B3 for each calendar day
    between *start_date* and *end_date* (weekends skipped automatically) and
    upserts the results into the ``asset_prices`` and ``companies`` tables.

    This stage is independent of the news pipeline and is designed to be run
    once to seed the historical price database.

    Args:
        start_date: First trading date to backfill (inclusive).
        end_date:   Last trading date to backfill (inclusive).
                    Defaults to today when not provided.
    """
    if end_date is None:
        end_date = datetime.date.today()

    logger.info("=" * 60)
    logger.info(
        f"Stage: BACKFILL — fetching all B3 prices from {start_date} to {end_date}"
    )
    logger.info("=" * 60)

    db = NewsDatabase()
    market_db = MarketDatabase(db_path=db.db_path)

    price_records = fetch_all_prices_range(start_date=start_date, end_date=end_date)

    if not price_records:
        logger.warning("No price records returned — check date range and network access")
        return

    prices_written = market_db.upsert_prices(price_records)
    logger.info(f"Stored {prices_written} price records in asset_prices")

    companies = extract_companies_from_prices(price_records)
    companies_written = market_db.upsert_companies(companies)
    logger.info(f"Stored {companies_written} company records in companies")

    logger.info("Stage BACKFILL complete\n")


def run_analytics() -> None:
    """
    Stage 4 – Analytics: compute news–price correlations.

    Reads enriched news and stored prices from the database and computes
    D0 / D+1 / D+5 price-movement metrics for every (news, ticker) pair.

    Both :func:`run_trusted` and :func:`run_prices` should have been
    executed before this stage so that the required data is available.
    """
    logger.info("=" * 60)
    logger.info("Stage: ANALYTICS — computing news–price correlations")
    logger.info("=" * 60)

    db = NewsDatabase()
    market_db = MarketDatabase(db_path=db.db_path)

    enriched_news = db.get_enriched_news()
    relevant_news = [
        n for n in enriched_news
        if n.get("sentiment") and _tickers_nonempty(n.get("tickers"))
    ]

    logger.info(f"Enriched news in DB: {len(enriched_news)}")
    logger.info(f"Records with sentiment + tickers: {len(relevant_news)}")

    if not relevant_news:
        logger.info("No enriched news with tickers — stage ANALYTICS complete")
        return

    correlation_records = compute_correlations(relevant_news, market_db)

    if correlation_records:
        written = market_db.upsert_correlations(correlation_records)
        logger.info(f"Stored {written} correlation records in news_price_correlation")
    else:
        logger.info("No correlation records computed (insufficient price data)")

    logger.info("Stage ANALYTICS complete\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(reprocess_existing: bool = False, fetch_market_data: bool = True) -> None:
    """
    Execute the full pipeline: raw → trusted → prices → analytics.

    This function acts as an orchestrator that calls each stage in order.
    It is preserved for backward compatibility.

    Args:
        reprocess_existing: Re-enrich all news records (trusted stage).
        fetch_market_data:  If False, skip the prices and analytics stages.
    """
    logger.info("=" * 60)
    logger.info("Starting full pipeline (raw → trusted → prices → analytics)")
    logger.info("=" * 60)

    try:
        run_raw()
        run_trusted(reprocess_all=reprocess_existing)
        if fetch_market_data:
            run_prices()
            run_analytics()

        logger.info("=" * 60)
        logger.info("Full pipeline completed successfully")
        logger.info("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the financial news ingestion pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage raw
  python main.py --stage trusted
  python main.py --stage trusted --reprocess-existing
  python main.py --stage prices
  python main.py --stage prices --tickers PETR4,VALE3 --date 2026-04-01
  python main.py --stage analytics
  python main.py --stage backfill
  python main.py --stage backfill --from 2025-01-01 --to 2025-12-31
  python main.py --stage all
  python main.py --stage all --no-market-data
""",
    )

    parser.add_argument(
        '--stage',
        choices=['raw', 'trusted', 'prices', 'analytics', 'backfill', 'all'],
        default='all',
        help=(
            'Pipeline stage to execute. '
            '"all" runs every stage in sequence (default).'
        ),
    )
    parser.add_argument(
        '--reprocess-existing',
        action='store_true',
        help='Re-enrich the entire news database (trusted / all stages only).',
    )
    parser.add_argument(
        '--no-market-data',
        action='store_true',
        help='Skip prices and analytics stages (all stage only).',
    )
    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        metavar='TICKER1,TICKER2',
        help='Comma-separated tickers for the prices stage.',
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        metavar='YYYY-MM-DD',
        help='Single trading date for the prices stage (auto-detected if omitted).',
    )
    parser.add_argument(
        '--from',
        dest='from_date',
        type=str,
        default='2025-01-01',
        metavar='YYYY-MM-DD',
        help='Start date for the backfill stage (default: 2025-01-01).',
    )
    parser.add_argument(
        '--to',
        dest='to_date',
        type=str,
        default=None,
        metavar='YYYY-MM-DD',
        help='End date for the backfill stage (default: today).',
    )

    args = parser.parse_args()

    _tickers_arg = (
        [t.strip() for t in args.tickers.split(',') if t.strip()]
        if args.tickers else None
    )
    _dates_arg = (
        [datetime.date.fromisoformat(args.date)]
        if args.date else None
    )
    _from_date = datetime.date.fromisoformat(args.from_date)
    _to_date = (
        datetime.date.fromisoformat(args.to_date) if args.to_date else None
    )

    if args.stage == 'raw':
        run_raw()
    elif args.stage == 'trusted':
        run_trusted(reprocess_all=args.reprocess_existing)
    elif args.stage == 'prices':
        run_prices(tickers=_tickers_arg, dates=_dates_arg)
    elif args.stage == 'analytics':
        run_analytics()
    elif args.stage == 'backfill':
        run_backfill(start_date=_from_date, end_date=_to_date)
    else:  # 'all'
        run_pipeline(
            reprocess_existing=args.reprocess_existing,
            fetch_market_data=not args.no_market_data,
        )
