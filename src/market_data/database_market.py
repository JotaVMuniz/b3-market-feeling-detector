"""
Database management for market data tables.

Adds three new tables to the existing news.db SQLite database:

- ``asset_prices``: daily OHLCV data per ticker.
- ``companies``: company/asset metadata derived from price files.
- ``news_price_correlation``: pre-computed price-movement metrics linked
  to news items with sentiment data.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CREATE_ASSET_PRICES = """
CREATE TABLE IF NOT EXISTS asset_prices (
    ticker    TEXT NOT NULL,
    date      TEXT NOT NULL,
    open      REAL,
    close     REAL,
    high      REAL,
    low       REAL,
    avg_price REAL,
    volume    REAL,
    PRIMARY KEY (ticker, date)
)
"""

_CREATE_COMPANIES = """
CREATE TABLE IF NOT EXISTS companies (
    ticker     TEXT PRIMARY KEY,
    name       TEXT,
    tipo_papel TEXT,
    isin       TEXT
)
"""

_CREATE_NEWS_PRICE_CORRELATION = """
CREATE TABLE IF NOT EXISTS news_price_correlation (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id    INTEGER NOT NULL,
    ticker     TEXT NOT NULL,
    news_date  TEXT NOT NULL,
    sentiment  TEXT,
    confidence REAL,
    d0_var     REAL,
    d1_var     REAL,
    d5_var     REAL,
    FOREIGN KEY (news_id) REFERENCES news(id),
    UNIQUE (news_id, ticker)
)
"""

_CREATE_IDX_ASSET_PRICES_TICKER = """
CREATE INDEX IF NOT EXISTS idx_asset_prices_ticker
ON asset_prices(ticker)
"""

_CREATE_IDX_CORRELATION_TICKER = """
CREATE INDEX IF NOT EXISTS idx_correlation_ticker
ON news_price_correlation(ticker)
"""

_CREATE_IDX_CORRELATION_NEWS = """
CREATE INDEX IF NOT EXISTS idx_correlation_news_id
ON news_price_correlation(news_id)
"""


class MarketDatabase:
    """Manages market-data tables inside the shared news.db database."""

    def __init__(self, db_path: str = "data/news.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_tables()

    @contextmanager
    def get_connection(self):
        """Yield a committed (or rolled-back) SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error(f"Database error: {exc}")
            raise
        finally:
            conn.close()

    def init_tables(self) -> None:
        """Create market-data tables and indexes if they do not exist."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for stmt in (
                    _CREATE_ASSET_PRICES,
                    _CREATE_COMPANIES,
                    _CREATE_NEWS_PRICE_CORRELATION,
                    _CREATE_IDX_ASSET_PRICES_TICKER,
                    _CREATE_IDX_CORRELATION_TICKER,
                    _CREATE_IDX_CORRELATION_NEWS,
                ):
                    cursor.execute(stmt)
            logger.info("Market-data tables initialised")
        except Exception as exc:
            logger.error(f"Error initialising market-data tables: {exc}")
            raise

    # ------------------------------------------------------------------
    # asset_prices
    # ------------------------------------------------------------------

    def upsert_prices(self, price_records: List[Dict]) -> int:
        """
        Insert or replace price records in ``asset_prices``.

        Args:
            price_records: Dicts with keys ticker, date, open, close,
                high, low, avg_price, volume.

        Returns:
            Number of rows written.
        """
        if not price_records:
            return 0

        written = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for rec in price_records:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO asset_prices
                            (ticker, date, open, close, high, low, avg_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec.get("ticker"),
                            rec.get("date"),
                            rec.get("open"),
                            rec.get("close"),
                            rec.get("high"),
                            rec.get("low"),
                            rec.get("avg_price"),
                            rec.get("volume"),
                        ),
                    )
                    written += 1
            logger.info(f"Upserted {written} price records")
        except Exception as exc:
            logger.error(f"Error upserting prices: {exc}")
        return written

    def get_price(self, ticker: str, date: str) -> Optional[Dict]:
        """
        Fetch a single price record for *ticker* on *date* (YYYY-MM-DD).

        Returns:
            Dict or None if no record exists.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM asset_prices WHERE ticker = ? AND date = ?",
                    (ticker, date),
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as exc:
            logger.error(f"Error fetching price for {ticker} on {date}: {exc}")
            return None

    def get_prices_for_ticker(self, ticker: str, date_from: str, date_to: str) -> List[Dict]:
        """
        Return all stored prices for *ticker* between *date_from* and *date_to*.

        Args:
            ticker: B3 ticker code.
            date_from: ISO date string (inclusive).
            date_to: ISO date string (inclusive).

        Returns:
            List of price dicts ordered by date ascending.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM asset_prices
                    WHERE ticker = ? AND date BETWEEN ? AND ?
                    ORDER BY date ASC
                    """,
                    (ticker, date_from, date_to),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error fetching prices for {ticker}: {exc}")
            return []

    # ------------------------------------------------------------------
    # companies
    # ------------------------------------------------------------------

    def upsert_companies(self, company_records: List[Dict]) -> int:
        """
        Insert or replace company metadata records.

        Args:
            company_records: Dicts with keys ticker, name, tipo_papel, isin.

        Returns:
            Number of rows written.
        """
        if not company_records:
            return 0

        written = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for rec in company_records:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO companies
                            (ticker, name, tipo_papel, isin)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            rec.get("ticker"),
                            rec.get("name"),
                            rec.get("tipo_papel"),
                            rec.get("isin"),
                        ),
                    )
                    written += 1
            logger.info(f"Upserted {written} company records")
        except Exception as exc:
            logger.error(f"Error upserting companies: {exc}")
        return written

    def get_all_companies(self) -> List[Dict]:
        """Return all company records ordered by ticker."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM companies ORDER BY ticker")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error fetching companies: {exc}")
            return []

    def get_known_tickers(self) -> set:
        """Return the set of tickers present in the companies table."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker FROM companies")
                return {row["ticker"] for row in cursor.fetchall()}
        except Exception as exc:
            logger.error(f"Error fetching known tickers: {exc}")
            return set()

    # ------------------------------------------------------------------
    # news_price_correlation
    # ------------------------------------------------------------------

    def upsert_correlations(self, correlation_records: List[Dict]) -> int:
        """
        Insert or replace news–price correlation records.

        Args:
            correlation_records: Dicts with keys news_id, ticker, news_date,
                sentiment, confidence, d0_var, d1_var, d5_var.

        Returns:
            Number of rows written.
        """
        if not correlation_records:
            return 0

        written = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for rec in correlation_records:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO news_price_correlation
                            (news_id, ticker, news_date, sentiment, confidence,
                             d0_var, d1_var, d5_var)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec.get("news_id"),
                            rec.get("ticker"),
                            rec.get("news_date"),
                            rec.get("sentiment"),
                            rec.get("confidence"),
                            rec.get("d0_var"),
                            rec.get("d1_var"),
                            rec.get("d5_var"),
                        ),
                    )
                    written += 1
            logger.info(f"Upserted {written} correlation records")
        except Exception as exc:
            logger.error(f"Error upserting correlations: {exc}")
        return written

    def get_correlations(
        self,
        ticker: Optional[str] = None,
        sentiment: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """
        Query correlation records with optional filters.

        Args:
            ticker: Filter to a specific ticker (optional).
            sentiment: Filter to a specific sentiment value (optional).
            limit: Maximum records to return.

        Returns:
            List of correlation dicts.
        """
        query = "SELECT * FROM news_price_correlation WHERE 1=1"
        params: List = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if sentiment:
            query += " AND sentiment = ?"
            params.append(sentiment)

        query += " ORDER BY news_date DESC LIMIT ?"
        params.append(limit)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error fetching correlations: {exc}")
            return []

    def get_correlations_with_news(self, limit: int = 1000) -> List[Dict]:
        """
        Return correlations joined with news title and source.

        Args:
            limit: Maximum records to return.

        Returns:
            List of dicts combining correlation and news fields.
        """
        query = """
            SELECT
                c.id,
                c.news_id,
                c.ticker,
                c.news_date,
                c.sentiment,
                c.confidence,
                c.d0_var,
                c.d1_var,
                c.d5_var,
                n.title,
                n.source,
                n.url
            FROM news_price_correlation c
            JOIN news n ON n.id = c.news_id
            ORDER BY c.news_date DESC
            LIMIT ?
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error fetching correlations with news: {exc}")
            return []
