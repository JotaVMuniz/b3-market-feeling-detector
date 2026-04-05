"""
Tests for the main pipeline orchestration module.
"""

from unittest.mock import Mock, patch

from main import run_pipeline


def test_run_pipeline_reprocess_existing_updates_database(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    sample_news = [
        {
            "title": "Empresa anuncia lucro",
            "summary": "Resultado trimestral positivo.",
            "link": "https://example.com/news1",
            "published_at": "2026-03-26T10:00:00",
            "source": "TestSource",
            "collected_at": "2026-03-26T11:00:00"
        }
    ]

    cleaned_news = [
        {
            "title": "Empresa anuncia lucro",
            "summary": "Resultado trimestral positivo.",
            "link": "https://example.com/news1",
            "published_at": "2026-03-26T10:00:00",
            "source": "TestSource",
            "collected_at": "2026-03-26T11:00:00"
        }
    ]

    enrichment_result = [
        {
            "is_relevant": True,
            "sentiment": "positivo",
            "confidence": 0.9,
            "segments": ["bancos"],
            "tickers": ["ITUB4"]
        }
    ]

    mock_db = Mock()
    mock_db.get_news_count.return_value = 0
    mock_db.insert_news.return_value = 1
    mock_db.get_all_news.return_value = [
        {
            "title": "Empresa anuncia lucro",
            "content": "Resultado trimestral positivo.",
            "source": "TestSource",
            "published_at": "2026-03-26T10:00:00",
            "url": "https://example.com/news1",
            "collected_at": "2026-03-26T11:00:00"
        }
    ]
    mock_db.update_news_batch.return_value = 1

    with patch("main.get_sources", return_value=["dummy_source"]), \
         patch("main.fetch_all_news", return_value=sample_news), \
         patch("main.deduplicate_news", return_value=sample_news), \
         patch("main.clean_news_batch", return_value=cleaned_news), \
         patch("main.validate_news_entry", return_value=True), \
         patch("main.enrich_batch", return_value=enrichment_result), \
         patch("main.save_raw_news", return_value="data/raw/news.json"), \
         patch("main.NewsDatabase", return_value=mock_db):
        run_pipeline(reprocess_existing=True)

    mock_db.insert_news.assert_called_once()
    mock_db.get_all_news.assert_called_once()
    mock_db.update_news_batch.assert_called_once()
