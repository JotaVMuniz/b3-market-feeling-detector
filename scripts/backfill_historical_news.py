"""
Backfill historical news using Wayback Machine snapshots of the RSS feeds.

RSS feeds only expose the *current* items (~10-100 latest), with no native
historical archive. However, the Internet Archive's Wayback Machine has been
periodically crawling these same feed URLs for years, which means each
snapshot is effectively a time capsule of "whatever was in the feed on that
date". By fetching many snapshots across the desired date range, we can
reconstruct a much richer historical news dataset than a single live fetch
allows.

This reuses the exact same parsing / cleaning / insertion code as the live
`raw` stage (src.ingestion.fetch_news, src.processing.clean_news,
src.storage.database) so that backfilled and live-collected news are
indistinguishable in the database. The only difference is that this script
does NOT apply the per-source checkpoint used by `main.py --stage raw`
(which is designed to skip old news and would otherwise reject every
historical item we are deliberately trying to insert).

Usage:
    python scripts/backfill_historical_news.py --from 2025-01-01 --to 2026-08-22
"""

import argparse
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.sources import get_sources
from src.ingestion.fetch_news import fetch_feed, deduplicate_news
from src.processing.clean_news import clean_news_batch, validate_news_entry
from src.storage.database import NewsDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_historical_news")

CDX_API = "http://web.archive.org/cdx/search/cdx"
WAYBACK_RAW = "http://web.archive.org/web/{timestamp}id_/{url}"


def get_snapshots(url: str, date_from: str, date_to: str) -> List[str]:
    """Return a list of Wayback Machine timestamps (YYYYMMDDhhmmss) for *url*."""
    params = {
        "url": url,
        "output": "json",
        "from": date_from.replace("-", ""),
        "to": date_to.replace("-", ""),
        "filter": "statuscode:200",
        "collapse": "timestamp:6",  # at most one snapshot per day (YYYYMMDD)
        "limit": 5000,
    }
    try:
        resp = requests.get(CDX_API, params=params, timeout=40)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        logger.warning(f"Could not query CDX API for {url}: {e}")
        return []
    if len(rows) <= 1:
        return []
    return [row[1] for row in rows[1:]]


def fetch_historical_news(date_from: str, date_to: str, delay: float = 0.5) -> List[Dict]:
    """Fetch news entries from Wayback snapshots of every configured RSS source."""
    all_news: List[Dict] = []

    for source in get_sources():
        snapshots = get_snapshots(source.url, date_from, date_to)
        logger.info(f"{source.name}: {len(snapshots)} snapshot(s) found between {date_from} and {date_to}")

        for ts in snapshots:
            archive_url = WAYBACK_RAW.format(timestamp=ts, url=source.url)
            entries = fetch_feed(archive_url, source.name)
            snap_date = datetime.strptime(ts[:8], "%Y%m%d").date().isoformat()
            logger.info(f"  snapshot {snap_date}: {len(entries)} entries")
            all_news.extend(entries)
            time.sleep(delay)

    return all_news


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", default="2025-01-01")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    parser.add_argument("--delay", type=float, default=0.5,
                         help="Seconds to sleep between snapshot requests (be polite to archive.org)")
    args = parser.parse_args()

    logger.info(f"Backfilling historical news from {args.date_from} to {args.date_to} via Wayback Machine")

    raw_news = fetch_historical_news(args.date_from, args.date_to, delay=args.delay)
    logger.info(f"Total raw entries across all snapshots: {len(raw_news)}")

    deduplicated = deduplicate_news(raw_news)
    logger.info(f"After in-batch deduplication: {len(deduplicated)}")

    cleaned = clean_news_batch(deduplicated)
    valid_news = [n for n in cleaned if validate_news_entry(n)]
    logger.info(f"Valid entries after cleaning: {len(valid_news)}")

    if not valid_news:
        logger.info("Nothing to insert.")
        return

    # No checkpoint filter here on purpose — we WANT old news.
    db = NewsDatabase()
    inserted = db.insert_news(valid_news)
    logger.info(f"Inserted {inserted} new historical records into news table "
                f"({len(valid_news) - inserted} were already present, skipped by URL uniqueness).")


if __name__ == "__main__":
    main()
