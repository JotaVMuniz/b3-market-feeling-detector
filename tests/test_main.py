"""
Tests for the main pipeline orchestration module.
"""

import datetime
from unittest.mock import Mock, patch

from main import (
    run_analytics,
    run_indicators,
    run_pipeline,
    run_prices,
    run_raw,
    run_trusted,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SAMPLE_NEWS = [
    {
        "title": "Empresa anuncia lucro",
        "summary": "Resultado trimestral positivo.",
        "link": "https://example.com/news1",
        "published_at": "2026-03-26T10:00:00",
        "source": "TestSource",
        "collected_at": "2026-03-26T11:00:00",
    }
]

_CLEANED_NEWS = [
    {
        "title": "Empresa anuncia lucro",
        "summary": "Resultado trimestral positivo.",
        "link": "https://example.com/news1",
        "published_at": "2026-03-26T10:00:00",
        "source": "TestSource",
        "collected_at": "2026-03-26T11:00:00",
    }
]

_ENRICHMENT_RESULT = [
    {
        "is_relevant": True,
        "market_relevance": 0.8,
        "sentiment": "positivo",
        "confidence": 0.9,
        "segments": ["bancos"],
        "tickers": ["ITUB4"],
    }
]

_DB_NEWS_ROW = {
    "id": 1,
    "title": "Empresa anuncia lucro",
    "content": "Resultado trimestral positivo.",
    "source": "TestSource",
    "published_at": "2026-03-26T10:00:00",
    "url": "https://example.com/news1",
    "collected_at": "2026-03-26T11:00:00",
    "sentiment": None,
    "confidence": None,
    "segments": None,
    "tickers": None,
}


def _make_mock_db(**overrides):
    mock_db = Mock()
    mock_db.db_path = "data/news.db"
    mock_db.get_news_count.return_value = 0
    mock_db.insert_news.return_value = 1
    mock_db.get_all_news.return_value = [_DB_NEWS_ROW]
    mock_db.get_news_without_enrichment.return_value = [_DB_NEWS_ROW]
    mock_db.get_enriched_news.return_value = []
    mock_db.update_news_batch.return_value = 1
    mock_db.get_latest_published_at_by_source.return_value = None
    for k, v in overrides.items():
        setattr(mock_db, k, v)
    return mock_db


def _make_mock_market_db():
    m = Mock()
    m.get_known_tickers.return_value = set()
    m.get_ibrx_tickers.return_value = []
    m.get_tickers_with_prices.return_value = set()
    m.upsert_prices.return_value = 0
    m.upsert_companies.return_value = 0
    m.upsert_correlations.return_value = 0
    m.upsert_indicators.return_value = 0
    m.upsert_composite_index.return_value = 0
    m.upsert_ibrx_tickers.return_value = 0
    m.get_indicators.return_value = []
    m.get_ingested_price_dates.return_value = set()
    m.get_latest_indicator_date.return_value = None
    return m


# ---------------------------------------------------------------------------
# run_pipeline (backward-compatibility test)
# ---------------------------------------------------------------------------

def test_run_pipeline_reprocess_existing_updates_database(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_db = _make_mock_db()
    mock_market_db = _make_mock_market_db()

    with patch("main.get_sources", return_value=["dummy_source"]), \
         patch("main.fetch_all_news", return_value=_SAMPLE_NEWS), \
         patch("main.deduplicate_news", return_value=_SAMPLE_NEWS), \
         patch("main.clean_news_batch", return_value=_CLEANED_NEWS), \
         patch("main.validate_news_entry", return_value=True), \
         patch("main.enrich_batch", return_value=_ENRICHMENT_RESULT), \
         patch("main.save_raw_news", return_value="data/raw/news.json"), \
         patch("main.NewsDatabase", return_value=mock_db), \
         patch("main.MarketDatabase", return_value=mock_market_db), \
         patch("main.fetch_prices_for_tickers", return_value=[]), \
         patch("main.fetch_market_indicators_range", return_value={}), \
         patch("main.fetch_bcb_indicators", return_value=[]), \
         patch("main.indicators_to_raw_records", return_value=[]), \
         patch("main.compute_composite_index", return_value=[]), \
         patch("main.compute_correlations", return_value=[]):
        run_pipeline(reprocess_existing=True)

    mock_db.insert_news.assert_called_once()
    # get_all_news is called by run_trusted (reprocess_all=True) and run_prices
    mock_db.get_all_news.assert_called()
    mock_db.update_news_batch.assert_called_once()


# ---------------------------------------------------------------------------
# run_raw
# ---------------------------------------------------------------------------

def test_run_raw_inserts_news():
    mock_db = _make_mock_db()

    with patch("main.get_sources", return_value=["dummy_source"]), \
         patch("main.fetch_all_news", return_value=_SAMPLE_NEWS), \
         patch("main.deduplicate_news", return_value=_SAMPLE_NEWS), \
         patch("main.clean_news_batch", return_value=_CLEANED_NEWS), \
         patch("main.validate_news_entry", return_value=True), \
         patch("main.save_raw_news", return_value="data/raw/news.json"), \
         patch("main.NewsDatabase", return_value=mock_db):
        run_raw()

    mock_db.insert_news.assert_called_once_with(_CLEANED_NEWS)


def test_run_raw_exits_early_when_no_news():
    mock_db = _make_mock_db()

    with patch("main.get_sources", return_value=[]), \
         patch("main.fetch_all_news", return_value=[]), \
         patch("main.NewsDatabase", return_value=mock_db):
        run_raw()

    mock_db.insert_news.assert_not_called()


# ---------------------------------------------------------------------------
# run_trusted
# ---------------------------------------------------------------------------

def test_run_trusted_enriches_unenriched_news(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_db = _make_mock_db()

    with patch("main.enrich_batch", return_value=_ENRICHMENT_RESULT), \
         patch("main.NewsDatabase", return_value=mock_db):
        run_trusted(reprocess_all=False)

    mock_db.get_news_without_enrichment.assert_called_once()
    mock_db.update_news_batch.assert_called_once()


def test_run_trusted_reprocess_all_uses_get_all_news(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_db = _make_mock_db()

    with patch("main.enrich_batch", return_value=_ENRICHMENT_RESULT), \
         patch("main.NewsDatabase", return_value=mock_db):
        run_trusted(reprocess_all=True)

    mock_db.get_all_news.assert_called_once()
    mock_db.get_news_without_enrichment.assert_not_called()
    mock_db.update_news_batch.assert_called_once()


def test_run_trusted_skips_when_nothing_to_enrich(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_db = _make_mock_db()
    mock_db.get_news_without_enrichment.return_value = []

    with patch("main.enrich_batch") as mock_enrich, \
         patch("main.NewsDatabase", return_value=mock_db):
        run_trusted(reprocess_all=False)

    mock_enrich.assert_not_called()
    mock_db.update_news_batch.assert_not_called()


# ---------------------------------------------------------------------------
# run_prices
# ---------------------------------------------------------------------------

def test_run_prices_skips_when_no_tickers():
    mock_db = _make_mock_db()
    mock_db.get_all_news.return_value = []  # no news → no tickers
    mock_market_db = _make_mock_market_db()

    with patch("main.NewsDatabase", return_value=mock_db), \
         patch("main.MarketDatabase", return_value=mock_market_db), \
         patch("main.fetch_prices_for_tickers") as mock_fetch:
        run_prices()

    mock_fetch.assert_not_called()


def test_run_prices_uses_explicit_tickers_and_dates():
    mock_db = _make_mock_db()
    mock_market_db = _make_mock_market_db()

    with patch("main.NewsDatabase", return_value=mock_db), \
         patch("main.MarketDatabase", return_value=mock_market_db), \
         patch("main.fetch_prices_for_tickers", return_value=[]) as mock_fetch, \
         patch("main.extract_companies_from_prices", return_value=[]):
        run_prices(
            tickers=["PETR4", "VALE3"],
            dates=[datetime.date(2026, 3, 26)],
        )

    mock_fetch.assert_called_once_with(
        tickers=["PETR4", "VALE3"],
        dates=[datetime.date(2026, 3, 26)],
    )
    # When explicit tickers/dates are supplied, DB should not be queried
    mock_db.get_all_news.assert_not_called()


def test_run_prices_stores_price_and_company_records():
    mock_db = _make_mock_db()
    mock_market_db = _make_mock_market_db()

    price_records = [
        {
            "ticker": "PETR4",
            "date": "2026-03-26",
            "open": 35.0,
            "close": 36.0,
            "high": 36.5,
            "low": 34.5,
            "avg_price": 35.5,
            "volume": 1_000_000.0,
            "nome_pregao": "PETROBRAS PN",
            "tipo_papel": "PN",
            "codigo_isin": "BRPETRACNPR6",
        }
    ]

    with patch("main.NewsDatabase", return_value=mock_db), \
         patch("main.MarketDatabase", return_value=mock_market_db), \
         patch("main.fetch_prices_for_tickers", return_value=price_records), \
         patch("main.extract_companies_from_prices", return_value=[{"ticker": "PETR4"}]):
        run_prices(
            tickers=["PETR4"],
            dates=[datetime.date(2026, 3, 26)],
        )

    mock_market_db.upsert_prices.assert_called_once_with(price_records)
    mock_market_db.upsert_companies.assert_called_once()


# ---------------------------------------------------------------------------
# run_analytics
# ---------------------------------------------------------------------------

def test_run_analytics_skips_when_no_enriched_news():
    mock_db = _make_mock_db()
    mock_db.get_enriched_news.return_value = []
    mock_market_db = _make_mock_market_db()

    with patch("main.NewsDatabase", return_value=mock_db), \
         patch("main.MarketDatabase", return_value=mock_market_db), \
         patch("main.compute_correlations") as mock_corr:
        run_analytics()

    mock_corr.assert_not_called()


def test_run_analytics_computes_and_stores_correlations():
    enriched_row = {
        "id": 1,
        "title": "Empresa anuncia lucro",
        "content": "Resultado trimestral positivo.",
        "url": "https://example.com/news1",
        "published_at": "2026-03-26T10:00:00",
        "sentiment": "positivo",
        "confidence": 0.9,
        "tickers": '["ITUB4"]',
    }

    mock_db = _make_mock_db()
    mock_db.get_enriched_news.return_value = [enriched_row]
    mock_market_db = _make_mock_market_db()

    corr_records = [
        {
            "news_id": 1,
            "ticker": "ITUB4",
            "news_date": "2026-03-26",
            "sentiment": "positivo",
            "confidence": 0.9,
            "d0_var": 0.02,
            "d1_var": 0.01,
            "d5_var": 0.05,
        }
    ]

    with patch("main.NewsDatabase", return_value=mock_db), \
         patch("main.MarketDatabase", return_value=mock_market_db), \
         patch("main.compute_correlations", return_value=corr_records):
        run_analytics()

    mock_market_db.upsert_correlations.assert_called_once_with(corr_records)


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestRawCheckpoint:
    """Checkpoint logic for the raw (RSS news) ingestion stage."""

    def test_filters_already_ingested_articles(self):
        """Articles published on or before the checkpoint date are skipped."""
        mock_db = _make_mock_db()
        # Latest stored article for TestSource is 2026-03-26
        mock_db.get_latest_published_at_by_source.return_value = "2026-03-26T10:00:00"
        mock_db.insert_news.return_value = 0

        with patch("main.get_sources", return_value=["dummy"]), \
             patch("main.fetch_all_news", return_value=_SAMPLE_NEWS), \
             patch("main.deduplicate_news", return_value=_SAMPLE_NEWS), \
             patch("main.clean_news_batch", return_value=_CLEANED_NEWS), \
             patch("main.validate_news_entry", return_value=True), \
             patch("main.save_raw_news", return_value="data/raw/news.json"), \
             patch("main.NewsDatabase", return_value=mock_db):
            run_raw()

        # All articles were filtered by the checkpoint, so insert should not be called
        mock_db.insert_news.assert_not_called()

    def test_passes_new_articles_through(self):
        """Articles published after the checkpoint date are inserted."""
        mock_db = _make_mock_db()
        # Checkpoint is earlier than the sample news published_at
        mock_db.get_latest_published_at_by_source.return_value = "2026-03-20T00:00:00"

        with patch("main.get_sources", return_value=["dummy"]), \
             patch("main.fetch_all_news", return_value=_SAMPLE_NEWS), \
             patch("main.deduplicate_news", return_value=_SAMPLE_NEWS), \
             patch("main.clean_news_batch", return_value=_CLEANED_NEWS), \
             patch("main.validate_news_entry", return_value=True), \
             patch("main.save_raw_news", return_value="data/raw/news.json"), \
             patch("main.NewsDatabase", return_value=mock_db):
            run_raw()

        mock_db.insert_news.assert_called_once()

    def test_no_checkpoint_inserts_all(self):
        """When no checkpoint exists (first run), all articles are inserted."""
        mock_db = _make_mock_db()
        mock_db.get_latest_published_at_by_source.return_value = None

        with patch("main.get_sources", return_value=["dummy"]), \
             patch("main.fetch_all_news", return_value=_SAMPLE_NEWS), \
             patch("main.deduplicate_news", return_value=_SAMPLE_NEWS), \
             patch("main.clean_news_batch", return_value=_CLEANED_NEWS), \
             patch("main.validate_news_entry", return_value=True), \
             patch("main.save_raw_news", return_value="data/raw/news.json"), \
             patch("main.NewsDatabase", return_value=mock_db):
            run_raw()

        mock_db.insert_news.assert_called_once()


class TestPriceCheckpoint:
    """Checkpoint logic for the prices stage."""

    def test_skips_already_ingested_dates(self):
        """Dates already in asset_prices are removed from the fetch list."""
        mock_db = _make_mock_db()
        mock_db.get_all_news.return_value = []  # will trigger fallback dates
        mock_market_db = _make_mock_market_db()

        target_date = datetime.date(2026, 3, 26)
        mock_market_db.get_known_tickers.return_value = {"PETR4"}
        mock_market_db.get_ingested_price_dates.return_value = {target_date.isoformat()}

        with patch("main.NewsDatabase", return_value=mock_db), \
             patch("main.MarketDatabase", return_value=mock_market_db), \
             patch("main.fetch_prices_for_tickers") as mock_fetch:
            run_prices(tickers=["PETR4"])  # dates auto-computed

        # fetch should be called only with dates NOT in the ingested set
        if mock_fetch.called:
            called_dates = mock_fetch.call_args[1]["dates"]
            assert target_date not in called_dates

    def test_explicit_dates_bypass_checkpoint(self):
        """When dates are explicitly passed, the checkpoint is not applied."""
        mock_db = _make_mock_db()
        mock_market_db = _make_mock_market_db()
        target_date = datetime.date(2026, 3, 26)
        # Even though this date is "ingested", explicit dates bypass checkpoint
        mock_market_db.get_ingested_price_dates.return_value = {target_date.isoformat()}

        with patch("main.NewsDatabase", return_value=mock_db), \
             patch("main.MarketDatabase", return_value=mock_market_db), \
             patch("main.fetch_prices_for_tickers", return_value=[]) as mock_fetch, \
             patch("main.extract_companies_from_prices", return_value=[]):
            run_prices(tickers=["PETR4"], dates=[target_date])

        mock_fetch.assert_called_once_with(tickers=["PETR4"], dates=[target_date])


class TestIndicatorCheckpoint:
    """Checkpoint logic for the indicators stage."""

    def test_advances_start_date_when_checkpoint_exists(self):
        """start_date is advanced to (checkpoint + 1 day) when auto-computed."""
        mock_db = _make_mock_db()
        mock_market_db = _make_mock_market_db()
        mock_market_db.get_latest_indicator_date.return_value = "2026-04-05"

        with patch("main.NewsDatabase", return_value=mock_db), \
             patch("main.MarketDatabase", return_value=mock_market_db), \
             patch("main.fetch_market_indicators_range", return_value={}) as mock_b3, \
             patch("main.fetch_bcb_indicators", return_value=[]), \
             patch("main.indicators_to_raw_records", return_value=[]), \
             patch("main.compute_composite_index", return_value=[]):
            run_indicators()  # start_date=None → auto-computed → checkpoint applied

        if mock_b3.called:
            actual_start = mock_b3.call_args[1].get("start_date") or mock_b3.call_args[0][0]
            assert actual_start >= datetime.date(2026, 4, 6)

    def test_explicit_start_date_bypasses_checkpoint(self):
        """When start_date is explicitly provided, the checkpoint is ignored."""
        mock_db = _make_mock_db()
        mock_market_db = _make_mock_market_db()
        mock_market_db.get_latest_indicator_date.return_value = "2026-04-05"

        explicit_start = datetime.date(2025, 1, 1)

        with patch("main.NewsDatabase", return_value=mock_db), \
             patch("main.MarketDatabase", return_value=mock_market_db), \
             patch("main.fetch_market_indicators_range", return_value={}) as mock_b3, \
             patch("main.fetch_bcb_indicators", return_value=[]), \
             patch("main.indicators_to_raw_records", return_value=[]), \
             patch("main.compute_composite_index", return_value=[]):
            run_indicators(start_date=explicit_start)

        mock_b3.assert_called_once()
        actual_start = mock_b3.call_args[1].get("start_date") or mock_b3.call_args[0][0]
        assert actual_start == explicit_start

    def test_skips_fetch_when_fully_up_to_date(self):
        """When the checkpoint equals today, no fetch is performed."""
        mock_db = _make_mock_db()
        mock_market_db = _make_mock_market_db()
        # Checkpoint is today → checkpoint+1 is tomorrow → start > end → skip
        mock_market_db.get_latest_indicator_date.return_value = (
            datetime.date.today().isoformat()
        )

        with patch("main.NewsDatabase", return_value=mock_db), \
             patch("main.MarketDatabase", return_value=mock_market_db), \
             patch("main.fetch_market_indicators_range") as mock_b3:
            run_indicators()

        mock_b3.assert_not_called()
