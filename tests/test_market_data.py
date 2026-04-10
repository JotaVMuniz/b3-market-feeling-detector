"""
Tests for the market data integration module.
"""

import datetime
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.market_data.database_market import MarketDatabase
from src.market_data.correlation import (
    _is_valid_ticker,
    _parse_tickers,
    _find_nearest_date,
    _nth_trading_date_after,
    _price_variation,
    _close_to_close_variation,
    compute_correlations,
)
from src.market_data.fetch_companies import extract_companies_from_prices


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path(tmp_path):
    """Return path to a temporary SQLite database file."""
    return str(tmp_path / "test_market.db")


@pytest.fixture
def market_db(tmp_db_path):
    """Create a fresh MarketDatabase for each test."""
    return MarketDatabase(db_path=tmp_db_path)


@pytest.fixture
def sample_prices():
    return [
        {
            "ticker": "PETR4",
            "date": "2024-11-15",
            "open": 38.00,
            "close": 39.50,
            "high": 40.00,
            "low": 37.50,
            "avg_price": 38.80,
            "volume": 1_000_000.0,
            "nome_pregao": "PETROBRAS",
            "tipo_papel": "ON  NM",
            "codigo_isin": "BRPETRACNPR6",
        },
        {
            "ticker": "PETR4",
            "date": "2024-11-18",
            "open": 39.00,
            "close": 40.00,
            "high": 40.50,
            "low": 38.80,
            "avg_price": 39.60,
            "volume": 900_000.0,
            "nome_pregao": "PETROBRAS",
            "tipo_papel": "ON  NM",
            "codigo_isin": "BRPETRACNPR6",
        },
        {
            "ticker": "VALE3",
            "date": "2024-11-15",
            "open": 60.00,
            "close": 58.00,
            "high": 61.00,
            "low": 57.50,
            "avg_price": 59.20,
            "volume": 2_000_000.0,
            "nome_pregao": "VALE",
            "tipo_papel": "ON  NM",
            "codigo_isin": "BRVALEACNOR0",
        },
    ]


@pytest.fixture
def sample_news_with_tickers():
    return [
        {
            "id": 1,
            "title": "Petrobras tem lucro recorde",
            "published_at": "2024-11-15T08:00:00",
            "tickers": json.dumps(["PETR4"]),
            "sentiment": "positivo",
            "confidence": 0.92,
        },
        {
            "id": 2,
            "title": "Vale reporta queda na produção",
            "published_at": "2024-11-15T09:30:00",
            "tickers": json.dumps(["VALE3"]),
            "sentiment": "negativo",
            "confidence": 0.85,
        },
        {
            "id": 3,
            "title": "Mercado sem tickers",
            "published_at": "2024-11-15T10:00:00",
            "tickers": json.dumps([]),
            "sentiment": "neutro",
            "confidence": 0.60,
        },
    ]


# ---------------------------------------------------------------------------
# Tests: MarketDatabase — asset_prices
# ---------------------------------------------------------------------------

class TestMarketDatabasePrices:
    def test_tables_created(self, market_db):
        with market_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row["name"] for row in cursor.fetchall()}
        assert "asset_prices" in tables
        assert "companies" in tables
        assert "news_price_correlation" in tables

    def test_upsert_prices(self, market_db, sample_prices):
        written = market_db.upsert_prices(sample_prices)
        assert written == len(sample_prices)

    def test_upsert_prices_idempotent(self, market_db, sample_prices):
        market_db.upsert_prices(sample_prices)
        written_again = market_db.upsert_prices(sample_prices)
        # INSERT OR REPLACE counts each row as written
        assert written_again == len(sample_prices)

    def test_get_price(self, market_db, sample_prices):
        market_db.upsert_prices(sample_prices)
        row = market_db.get_price("PETR4", "2024-11-15")
        assert row is not None
        assert row["ticker"] == "PETR4"
        assert row["close"] == pytest.approx(39.50)

    def test_get_price_missing(self, market_db):
        row = market_db.get_price("XXXX9", "2099-01-01")
        assert row is None

    def test_get_prices_for_ticker(self, market_db, sample_prices):
        market_db.upsert_prices(sample_prices)
        rows = market_db.get_prices_for_ticker("PETR4", "2024-11-01", "2024-11-30")
        assert len(rows) == 2
        assert rows[0]["date"] < rows[1]["date"]

    def test_upsert_empty(self, market_db):
        written = market_db.upsert_prices([])
        assert written == 0


# ---------------------------------------------------------------------------
# Tests: MarketDatabase — companies
# ---------------------------------------------------------------------------

class TestMarketDatabaseCompanies:
    def test_upsert_companies(self, market_db):
        records = [
            {"ticker": "PETR4", "name": "PETROBRAS", "tipo_papel": "ON  NM", "isin": "BRPETRACNPR6"},
            {"ticker": "VALE3", "name": "VALE", "tipo_papel": "ON  NM", "isin": "BRVALEACNOR0"},
        ]
        written = market_db.upsert_companies(records)
        assert written == 2

    def test_get_all_companies(self, market_db):
        records = [
            {"ticker": "PETR4", "name": "PETROBRAS", "tipo_papel": "ON  NM", "isin": "BRPETRACNPR6"},
        ]
        market_db.upsert_companies(records)
        companies = market_db.get_all_companies()
        assert len(companies) == 1
        assert companies[0]["ticker"] == "PETR4"

    def test_get_known_tickers(self, market_db):
        records = [
            {"ticker": "ITUB4", "name": "ITAUSA", "tipo_papel": "PN  N1", "isin": "BRITUBACNPR5"},
        ]
        market_db.upsert_companies(records)
        known = market_db.get_known_tickers()
        assert "ITUB4" in known

    def test_upsert_companies_empty(self, market_db):
        written = market_db.upsert_companies([])
        assert written == 0


# ---------------------------------------------------------------------------
# Tests: MarketDatabase — correlations
# ---------------------------------------------------------------------------

class TestMarketDatabaseCorrelations:
    def _setup_news_table(self, market_db):
        """Insert a minimal news row so the foreign key is satisfied."""
        with market_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    source TEXT NOT NULL,
                    published_at TEXT,
                    url TEXT NOT NULL UNIQUE,
                    collected_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                INSERT OR IGNORE INTO news (id, title, source, url, collected_at)
                VALUES (1, 'Test', 'TestSrc', 'http://example.com/1', '2024-11-15')
            """)
            cursor.execute("""
                INSERT OR IGNORE INTO news (id, title, source, url, collected_at)
                VALUES (2, 'Test2', 'TestSrc', 'http://example.com/2', '2024-11-15')
            """)

    def test_upsert_and_get_correlations(self, market_db):
        self._setup_news_table(market_db)
        records = [
            {
                "news_id": 1,
                "ticker": "PETR4",
                "news_date": "2024-11-15",
                "sentiment": "positivo",
                "confidence": 0.9,
                "d0_var": 0.03,
                "d1_var": 0.015,
                "d5_var": -0.01,
            }
        ]
        written = market_db.upsert_correlations(records)
        assert written == 1

        rows = market_db.get_correlations(ticker="PETR4")
        assert len(rows) == 1
        assert rows[0]["d0_var"] == pytest.approx(0.03)

    def test_get_correlations_by_sentiment(self, market_db):
        self._setup_news_table(market_db)
        records = [
            {
                "news_id": 1,
                "ticker": "PETR4",
                "news_date": "2024-11-15",
                "sentiment": "positivo",
                "confidence": 0.9,
                "d0_var": 0.03,
                "d1_var": None,
                "d5_var": None,
            },
            {
                "news_id": 2,
                "ticker": "VALE3",
                "news_date": "2024-11-15",
                "sentiment": "negativo",
                "confidence": 0.8,
                "d0_var": -0.02,
                "d1_var": None,
                "d5_var": None,
            },
        ]
        market_db.upsert_correlations(records)

        positivos = market_db.get_correlations(sentiment="positivo")
        assert len(positivos) == 1
        assert positivos[0]["ticker"] == "PETR4"

    def test_upsert_correlations_empty(self, market_db):
        written = market_db.upsert_correlations([])
        assert written == 0


# ---------------------------------------------------------------------------
# Tests: correlation helpers
# ---------------------------------------------------------------------------

class TestCorrelationHelpers:
    def test_is_valid_ticker(self):
        assert _is_valid_ticker("PETR4") is True
        assert _is_valid_ticker("VALE3") is True
        assert _is_valid_ticker("ITUB4") is True
        assert _is_valid_ticker("petr4") is False
        assert _is_valid_ticker("PETR") is False
        assert _is_valid_ticker("PETR44") is True   # 4 letters + 2 digits allowed
        assert _is_valid_ticker("") is False

    def test_parse_tickers_valid(self):
        result = _parse_tickers('["PETR4", "VALE3"]')
        assert result == ["PETR4", "VALE3"]

    def test_parse_tickers_empty(self):
        assert _parse_tickers("[]") == []
        assert _parse_tickers(None) == []
        assert _parse_tickers("") == []

    def test_parse_tickers_filters_invalid(self):
        result = _parse_tickers('["PETR4", "invalid", "123"]')
        assert result == ["PETR4"]

    def test_find_nearest_date_forward(self):
        dates = ["2024-11-15", "2024-11-18", "2024-11-19"]
        # Exact match
        assert _find_nearest_date("2024-11-15", dates, "forward") == "2024-11-15"
        # Weekend → next Monday
        assert _find_nearest_date("2024-11-16", dates, "forward") == "2024-11-18"

    def test_find_nearest_date_backward(self):
        dates = ["2024-11-15", "2024-11-18", "2024-11-19"]
        assert _find_nearest_date("2024-11-20", dates, "backward") == "2024-11-19"
        assert _find_nearest_date("2024-11-14", dates, "backward") is None

    def test_nth_trading_date_after(self):
        dates = ["2024-11-15", "2024-11-18", "2024-11-19", "2024-11-20", "2024-11-21", "2024-11-22"]
        # D+1 after 2024-11-15
        assert _nth_trading_date_after("2024-11-15", 1, dates) == "2024-11-18"
        # D+5 after 2024-11-15
        assert _nth_trading_date_after("2024-11-15", 5, dates) == "2024-11-22"
        # Not enough dates
        assert _nth_trading_date_after("2024-11-15", 10, dates) is None

    def test_price_variation(self):
        assert _price_variation(100.0, 105.0) == pytest.approx(0.05)
        assert _price_variation(100.0, 95.0) == pytest.approx(-0.05)
        assert _price_variation(0.0, 100.0) is None
        assert _price_variation(None, 100.0) is None
        assert _price_variation(100.0, None) is None

    def test_close_to_close_variation(self):
        assert _close_to_close_variation(100.0, 110.0) == pytest.approx(0.10)
        assert _close_to_close_variation(0.0, 100.0) is None


# ---------------------------------------------------------------------------
# Tests: compute_correlations integration
# ---------------------------------------------------------------------------

class TestComputeCorrelations:
    def test_computes_d0_d1(self, market_db, sample_news_with_tickers, sample_prices):
        # Add extra dates for D+1 and D+5
        extra_prices = []
        for i in range(1, 6):
            d = datetime.date(2024, 11, 18) + datetime.timedelta(days=i)
            if d.weekday() < 5:
                extra_prices.append({
                    "ticker": "PETR4",
                    "date": d.isoformat(),
                    "open": 39.0 + i,
                    "close": 40.0 + i,
                    "high": 41.0 + i,
                    "low": 38.0 + i,
                    "avg_price": 39.5 + i,
                    "volume": 800_000.0,
                    "nome_pregao": "PETROBRAS",
                    "tipo_papel": "ON  NM",
                    "codigo_isin": "BRPETRACNPR6",
                })
        market_db.upsert_prices(sample_prices + extra_prices)

        results = compute_correlations(sample_news_with_tickers[:1], market_db)
        assert len(results) == 1
        r = results[0]
        assert r["news_id"] == 1
        assert r["ticker"] == "PETR4"
        assert r["d0_var"] is not None
        assert r["sentiment"] == "positivo"

    def test_skips_news_without_tickers(self, market_db, sample_news_with_tickers, sample_prices):
        market_db.upsert_prices(sample_prices)
        # Only the news with empty tickers list
        no_ticker_news = [sample_news_with_tickers[2]]
        results = compute_correlations(no_ticker_news, market_db)
        assert results == []

    def test_skips_news_without_sentiment(self, market_db, sample_prices):
        market_db.upsert_prices(sample_prices)
        news = [{
            "id": 99,
            "title": "No sentiment",
            "published_at": "2024-11-15",
            "tickers": json.dumps(["PETR4"]),
            "sentiment": None,
            "confidence": None,
        }]
        results = compute_correlations(news, market_db)
        assert results == []

    def test_returns_none_when_no_prices(self, market_db, sample_news_with_tickers):
        # No prices inserted
        results = compute_correlations(sample_news_with_tickers, market_db)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: fetch_companies helpers
# ---------------------------------------------------------------------------

class TestExtractCompanies:
    def test_extracts_unique_companies(self, sample_prices):
        companies = extract_companies_from_prices(sample_prices)
        tickers = [c["ticker"] for c in companies]
        assert "PETR4" in tickers
        assert "VALE3" in tickers
        # Duplicate PETR4 rows should yield only one company entry
        assert tickers.count("PETR4") == 1

    def test_empty_input(self):
        assert extract_companies_from_prices([]) == []

    def test_fields_populated(self, sample_prices):
        companies = extract_companies_from_prices(sample_prices)
        petr = next(c for c in companies if c["ticker"] == "PETR4")
        assert petr["name"] == "PETROBRAS"
        assert petr["tipo_papel"] == "ON  NM"
        assert petr["isin"] == "BRPETRACNPR6"


# ---------------------------------------------------------------------------
# Tests: fetch_prices future-date guard
# ---------------------------------------------------------------------------

class TestFetchDailyPricesFutureDateGuard:
    def test_future_date_returns_empty_without_network_call(self):
        """fetch_daily_prices must return [] for future dates without attempting
        any network request (no B3 import / instantiation should occur)."""
        from src.market_data.fetch_prices import fetch_daily_prices

        future = datetime.date.today() + datetime.timedelta(days=1)

        with patch("src.market_data.fetch_prices.fetch_daily_prices") as _:
            # Call the real function, but patch mercados so any accidental
            # network call would raise to make the test fail
            pass

        # Test the real function directly – it must bail before importing B3
        with patch.dict("sys.modules", {"mercados.b3": None}):
            # Even if mercados is importable, future dates should short-circuit
            # before the import statement inside the function
            result = fetch_daily_prices(future)

        assert result == []

    def test_today_is_not_skipped(self):
        """Today's date should not be filtered by the future-date guard
        (even if the data isn't available yet, we attempt the request)."""
        from src.market_data import fetch_prices as fp

        today = datetime.date.today()
        # Patch B3 to avoid real network calls
        mock_b3_instance = MagicMock()
        mock_b3_instance.negociacao_bolsa.return_value = iter([])
        mock_b3_cls = MagicMock(return_value=mock_b3_instance)

        with patch.dict("sys.modules", {"mercados.b3": MagicMock(B3=mock_b3_cls)}):
            result = fp.fetch_daily_prices(today)

        assert result == []  # Empty because we mocked an empty iterator
        mock_b3_instance.negociacao_bolsa.assert_called_once_with("dia", today)

    def test_fetch_prices_for_tickers_skips_future_dates(self):
        """fetch_prices_for_tickers must not attempt fetches for future dates."""
        from src.market_data import fetch_prices as fp

        past_date = datetime.date(2024, 11, 15)
        future_date = datetime.date.today() + datetime.timedelta(days=3)

        fetched_dates = []

        def mock_fetch_daily(date):
            fetched_dates.append(date)
            return []

        with patch.object(fp, "fetch_daily_prices", side_effect=mock_fetch_daily):
            fp.fetch_prices_for_tickers(["PETR4"], [past_date, future_date])

        assert future_date not in fetched_dates
        assert past_date in fetched_dates


# ---------------------------------------------------------------------------
# Checkpoint helpers (new methods on MarketDatabase)
# ---------------------------------------------------------------------------

class TestMarketDatabaseCheckpoints:
    """Tests for get_ingested_price_dates and get_latest_indicator_date."""

    def test_get_ingested_price_dates_returns_empty_set_when_no_data(self, market_db):
        """With no prices stored, the set should be empty."""
        result = market_db.get_ingested_price_dates()
        assert result == set()

    def test_get_ingested_price_dates_returns_stored_dates(self, market_db, sample_prices):
        """Dates from asset_prices should appear in the returned set."""
        market_db.upsert_prices(sample_prices)
        dates = market_db.get_ingested_price_dates()
        expected = {r["date"] for r in sample_prices}
        assert expected.issubset(dates)

    def test_get_ingested_price_dates_returns_distinct_dates(self, market_db):
        """Even if multiple tickers have prices on the same date, the date appears once."""
        prices = [
            {"ticker": "PETR4", "date": "2026-04-01", "open": 35.0, "close": 36.0,
             "high": 36.5, "low": 34.5, "avg_price": 35.5, "volume": 100.0},
            {"ticker": "VALE3", "date": "2026-04-01", "open": 60.0, "close": 61.0,
             "high": 62.0, "low": 59.0, "avg_price": 60.5, "volume": 200.0},
        ]
        market_db.upsert_prices(prices)
        dates = market_db.get_ingested_price_dates()
        assert dates == {"2026-04-01"}

    def test_get_latest_indicator_date_returns_none_when_empty(self, market_db):
        """With no indicators stored, None should be returned."""
        result = market_db.get_latest_indicator_date()
        assert result is None

    def test_get_latest_indicator_date_returns_max_date(self, market_db):
        """Should return the most recent date across all indicators."""
        records = [
            {"date": "2026-03-01", "indicator": "turnover", "value": 1_000_000.0},
            {"date": "2026-04-05", "indicator": "turnover", "value": 2_000_000.0},
            {"date": "2026-04-03", "indicator": "trin", "value": 1.2},
        ]
        market_db.upsert_indicators(records)
        result = market_db.get_latest_indicator_date()
        assert result == "2026-04-05"


# ---------------------------------------------------------------------------
# IBrX 100 — database methods
# ---------------------------------------------------------------------------

class TestIbrxTickersDatabase:
    def test_upsert_and_get(self, market_db):
        tickers = ["PETR4", "VALE3", "ITUB4"]
        written = market_db.upsert_ibrx_tickers(tickers, "2026-04-08")
        assert written == 3
        result = market_db.get_ibrx_tickers()
        assert sorted(result) == sorted(tickers)

    def test_empty_list_returns_zero(self, market_db):
        assert market_db.upsert_ibrx_tickers([], "2026-04-08") == 0

    def test_empty_db_returns_empty_list(self, market_db):
        assert market_db.get_ibrx_tickers() == []

    def test_upsert_replaces_previous_list(self, market_db):
        market_db.upsert_ibrx_tickers(["PETR4", "VALE3"], "2026-04-07")
        market_db.upsert_ibrx_tickers(["WEGE3", "BBAS3"], "2026-04-08")
        result = market_db.get_ibrx_tickers()
        assert "PETR4" not in result
        assert "WEGE3" in result

    def test_returns_sorted_alphabetically(self, market_db):
        market_db.upsert_ibrx_tickers(["VALE3", "ABEV3", "PETR4"], "2026-04-08")
        result = market_db.get_ibrx_tickers()
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# IBrX 100 — fetch_ibrx100_tickers
# ---------------------------------------------------------------------------

class TestFetchIbrx100Tickers:
    from src.market_data.fetch_ibrx import _IBRX100_FALLBACK, fetch_ibrx100_tickers

    @patch("requests.get")
    def test_returns_tickers_from_api(self, mock_get):
        from src.market_data.fetch_ibrx import fetch_ibrx100_tickers
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"cod": "PETR4"},
                {"cod": "VALE3"},
                {"cod": "ITUB4"},
            ]
        }
        mock_get.return_value = mock_resp
        result = fetch_ibrx100_tickers()
        assert "PETR4" in result
        assert "VALE3" in result
        assert "ITUB4" in result

    @patch("requests.get")
    def test_falls_back_on_network_error(self, mock_get):
        from src.market_data.fetch_ibrx import fetch_ibrx100_tickers, _IBRX100_FALLBACK
        mock_get.side_effect = OSError("network error")
        result = fetch_ibrx100_tickers()
        assert result == list(_IBRX100_FALLBACK)

    @patch("requests.get")
    def test_falls_back_on_empty_results(self, mock_get):
        from src.market_data.fetch_ibrx import fetch_ibrx100_tickers, _IBRX100_FALLBACK
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_get.return_value = mock_resp
        result = fetch_ibrx100_tickers()
        assert result == list(_IBRX100_FALLBACK)

    @patch("requests.get")
    def test_tickers_are_uppercase(self, mock_get):
        from src.market_data.fetch_ibrx import fetch_ibrx100_tickers
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": [{"cod": "petr4"}, {"cod": "vale3"}]}
        mock_get.return_value = mock_resp
        result = fetch_ibrx100_tickers()
        assert all(t == t.upper() for t in result)

    @patch("requests.get")
    def test_strips_whitespace(self, mock_get):
        from src.market_data.fetch_ibrx import fetch_ibrx100_tickers
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": [{"cod": " PETR4 "}]}
        mock_get.return_value = mock_resp
        result = fetch_ibrx100_tickers()
        assert "PETR4" in result
