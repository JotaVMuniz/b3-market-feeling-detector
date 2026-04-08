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

_CREATE_SENTIMENT_INDICATORS = """
CREATE TABLE IF NOT EXISTS sentiment_indicators (
    date      TEXT NOT NULL,
    indicator TEXT NOT NULL,
    value     REAL,
    PRIMARY KEY (date, indicator)
)
"""

_CREATE_COMPOSITE_SENTIMENT_INDEX = """
CREATE TABLE IF NOT EXISTS composite_sentiment_index (
    date               TEXT PRIMARY KEY,
    score              REAL,
    label              TEXT,
    turnover_score     REAL,
    trin_score         REAL,
    put_call_score     REAL,
    pct_advancing_score REAL,
    cdi_score          REAL,
    consumer_confidence_score REAL,
    cds_score          REAL
)
"""

_CREATE_IDX_SENTIMENT_INDICATORS_DATE = """
CREATE INDEX IF NOT EXISTS idx_sentiment_indicators_date
ON sentiment_indicators(date)
"""

_CREATE_ASSET_FUNDAMENTALS = """
CREATE TABLE IF NOT EXISTS asset_fundamentals (
    ticker     TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      REAL,
    label      TEXT,
    updated_at TEXT,
    PRIMARY KEY (ticker, key)
)
"""

_CREATE_IDX_ASSET_FUNDAMENTALS_TICKER = """
CREATE INDEX IF NOT EXISTS idx_asset_fundamentals_ticker
ON asset_fundamentals(ticker)
"""

_CREATE_IBRX_TICKERS = """
CREATE TABLE IF NOT EXISTS ibrx_tickers (
    ticker     TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL
)
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
                    _CREATE_SENTIMENT_INDICATORS,
                    _CREATE_COMPOSITE_SENTIMENT_INDEX,
                    _CREATE_IDX_SENTIMENT_INDICATORS_DATE,
                    _CREATE_ASSET_FUNDAMENTALS,
                    _CREATE_IDX_ASSET_FUNDAMENTALS_TICKER,
                    _CREATE_IBRX_TICKERS,
                ):
                    cursor.execute(stmt)
            logger.info("Market-data tables initialised")
        except Exception as exc:
            logger.error(f"Error initialising market-data tables: {exc}")
            raise

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def get_ingested_price_dates(self) -> set:
        """
        Return the set of dates (``YYYY-MM-DD`` strings) that have at least
        one price record in ``asset_prices``.

        Used as a checkpoint so the prices stage skips dates that have
        already been fetched from B3, avoiding redundant API calls.

        Returns:
            Set of ISO date strings.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT date FROM asset_prices")
                return {row["date"] for row in cursor.fetchall()}
        except Exception as exc:
            logger.error("Error querying ingested price dates: %s", exc)
            return set()

    def get_latest_indicator_date(self) -> Optional[str]:
        """
        Return the most recent date present in ``sentiment_indicators``, or
        ``None`` when the table is empty.

        Used as a checkpoint so the indicators stage only fetches the delta
        (days after the last stored date) rather than re-fetching the full
        history on every run.

        Returns:
            ISO date string (``YYYY-MM-DD``) or ``None``.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT MAX(date) AS latest FROM sentiment_indicators"
                )
                row = cursor.fetchone()
                return row["latest"] if row and row["latest"] else None
        except Exception as exc:
            logger.error("Error querying latest indicator date: %s", exc)
            return None

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

    def get_tickers_with_prices(self) -> set:
        """Return the set of tickers that have at least one row in asset_prices.

        This is more reliable than ``get_known_tickers`` for the fundamentals
        stage because ``asset_prices`` only contains tickers that were
        successfully fetched from B3, whereas the ``companies`` table is
        populated from NLP-extracted mentions and may contain invalid codes
        (e.g. ``ITSAF130``) that do not exist on Yahoo Finance.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT ticker FROM asset_prices")
                return {row["ticker"] for row in cursor.fetchall()}
        except Exception as exc:
            logger.error(f"Error fetching tickers with prices: {exc}")
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

    # ------------------------------------------------------------------
    # sentiment_indicators
    # ------------------------------------------------------------------

    def upsert_indicators(self, records: List[Dict]) -> int:
        """
        Insert or replace rows in ``sentiment_indicators``.

        Args:
            records: Dicts with keys ``date``, ``indicator``, ``value``.

        Returns:
            Number of rows written.
        """
        if not records:
            return 0

        written = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for rec in records:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO sentiment_indicators
                            (date, indicator, value)
                        VALUES (?, ?, ?)
                        """,
                        (rec.get("date"), rec.get("indicator"), rec.get("value")),
                    )
                    written += 1
            logger.info(f"Upserted {written} sentiment indicator records")
        except Exception as exc:
            logger.error(f"Error upserting sentiment indicators: {exc}")
        return written

    def get_indicators(
        self,
        indicator: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict]:
        """
        Query rows from ``sentiment_indicators``.

        Args:
            indicator: Filter to a specific indicator name (optional).
            date_from: ISO date string lower bound (inclusive, optional).
            date_to:   ISO date string upper bound (inclusive, optional).

        Returns:
            List of indicator dicts ordered by date ascending.
        """
        query = "SELECT * FROM sentiment_indicators WHERE 1=1"
        params: List = []

        if indicator:
            query += " AND indicator = ?"
            params.append(indicator)
        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND date <= ?"
            params.append(date_to)

        query += " ORDER BY date ASC, indicator ASC"

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error fetching sentiment indicators: {exc}")
            return []

    # ------------------------------------------------------------------
    # composite_sentiment_index
    # ------------------------------------------------------------------

    def upsert_composite_index(self, records: List[Dict]) -> int:
        """
        Insert or replace rows in ``composite_sentiment_index``.

        Args:
            records: Dicts with keys ``date``, ``score``, ``label``, and
                optional per-indicator score keys.

        Returns:
            Number of rows written.
        """
        if not records:
            return 0

        written = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for rec in records:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO composite_sentiment_index
                            (date, score, label, turnover_score, trin_score,
                             put_call_score, pct_advancing_score, cdi_score,
                             consumer_confidence_score, cds_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec.get("date"),
                            rec.get("score"),
                            rec.get("label"),
                            rec.get("turnover_score"),
                            rec.get("trin_score"),
                            rec.get("put_call_score"),
                            rec.get("pct_advancing_score"),
                            rec.get("cdi_score"),
                            rec.get("consumer_confidence_score"),
                            rec.get("cds_score"),
                        ),
                    )
                    written += 1
            logger.info(f"Upserted {written} composite index records")
        except Exception as exc:
            logger.error(f"Error upserting composite index: {exc}")
        return written

    def get_composite_index(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 365,
    ) -> List[Dict]:
        """
        Query rows from ``composite_sentiment_index``.

        Args:
            date_from: ISO date lower bound (inclusive, optional).
            date_to:   ISO date upper bound (inclusive, optional).
            limit:     Maximum number of rows to return.

        Returns:
            List of composite index dicts ordered by date ascending.
        """
        query = "SELECT * FROM composite_sentiment_index WHERE 1=1"
        params: List = []

        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND date <= ?"
            params.append(date_to)

        query += " ORDER BY date ASC LIMIT ?"
        params.append(limit)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error fetching composite index: {exc}")
            return []

    # ------------------------------------------------------------------
    # asset_fundamentals
    # ------------------------------------------------------------------

    def upsert_fundamentals(self, records: List[Dict]) -> int:
        """
        Insert or replace rows in ``asset_fundamentals``.

        Args:
            records: Dicts with keys ``ticker``, ``key``, ``value``,
                ``label``, ``updated_at``.

        Returns:
            Number of rows written.
        """
        if not records:
            return 0

        written = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for rec in records:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO asset_fundamentals
                            (ticker, key, value, label, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            rec.get("ticker"),
                            rec.get("key"),
                            rec.get("value"),
                            rec.get("label"),
                            rec.get("updated_at"),
                        ),
                    )
                    written += 1
            logger.info(f"Upserted {written} fundamental records")
        except Exception as exc:
            logger.error(f"Error upserting fundamentals: {exc}")
        return written

    def get_fundamentals(self, ticker: str) -> List[Dict]:
        """
        Return all stored fundamental indicator rows for *ticker*.

        Args:
            ticker: B3 ticker code or ``"__MACRO__"`` for macro indicators.

        Returns:
            List of fundamental dicts ordered by key.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM asset_fundamentals WHERE ticker = ? ORDER BY key",
                    (ticker,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error fetching fundamentals for {ticker}: {exc}")
            return []

    def get_fundamentals_updated_at(self, ticker: str) -> Optional[str]:
        """
        Return the most recent ``updated_at`` value for *ticker*'s
        fundamental data, or ``None`` when no data is stored.

        Args:
            ticker: B3 ticker code.

        Returns:
            ISO date string or ``None``.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT MAX(updated_at) AS latest FROM asset_fundamentals "
                    "WHERE ticker = ?",
                    (ticker,),
                )
                row = cursor.fetchone()
                return row["latest"] if row and row["latest"] else None
        except Exception as exc:
            logger.error(
                f"Error fetching fundamentals updated_at for {ticker}: {exc}"
            )
            return None

    # ------------------------------------------------------------------
    # ibrx_tickers
    # ------------------------------------------------------------------

    def upsert_ibrx_tickers(self, tickers: List[str], updated_at: str) -> int:
        """
        Replace the stored IBrX 100 ticker list with *tickers*.

        Existing rows are deleted first so the table always reflects the
        current index composition exactly.

        Args:
            tickers:    List of B3 ticker codes.
            updated_at: ISO date string (``YYYY-MM-DD``) of the fetch date.

        Returns:
            Number of rows written.
        """
        if not tickers:
            return 0
        written = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM ibrx_tickers")
                for ticker in tickers:
                    cursor.execute(
                        "INSERT OR REPLACE INTO ibrx_tickers (ticker, updated_at) "
                        "VALUES (?, ?)",
                        (ticker, updated_at),
                    )
                    written += 1
            logger.info("Stored %d IBrX 100 tickers in ibrx_tickers", written)
        except Exception as exc:
            logger.error("Error upserting IBrX 100 tickers: %s", exc)
        return written

    def get_ibrx_tickers(self) -> List[str]:
        """
        Return the stored IBrX 100 ticker list ordered alphabetically.

        Returns an empty list when the table has never been populated (i.e.
        ``run_ibrx_tickers`` has not been executed yet for this database).

        Returns:
            List of B3 ticker codes.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker FROM ibrx_tickers ORDER BY ticker")
                return [row["ticker"] for row in cursor.fetchall()]
        except Exception as exc:
            logger.error("Error fetching IBrX 100 tickers: %s", exc)
            return []
