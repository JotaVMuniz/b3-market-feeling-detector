"""
Tests for the main pipeline orchestration module.
"""

import datetime
from unittest.mock import Mock, patch

from main import run_analytics, run_pipeline, run_prices, run_raw, run_trusted


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
    for k, v in overrides.items():
        setattr(mock_db, k, v)
    return mock_db


def _make_mock_market_db():
    m = Mock()
    m.get_known_tickers.return_value = set()
    m.upsert_prices.return_value = 0
    m.upsert_companies.return_value = 0
    m.upsert_correlations.return_value = 0
    m.upsert_indicators.return_value = 0
    m.upsert_composite_index.return_value = 0
    m.get_indicators.return_value = []
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
         patch("main.fetch_tweets", return_value=[]), \
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
