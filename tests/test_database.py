"""
Tests for the database module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.storage.database import NewsDatabase


class TestNewsDatabase:
    """Tests for the NewsDatabase class."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_path = temp_db.name
        temp_db.close()
        
        db_instance = NewsDatabase(db_path=db_path)
        yield db_instance
        
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
    
    def test_database_initialization(self, db):
        """Test database initialization."""
        assert Path(db.db_path).exists()
    
    def test_insert_news(self, db, sample_news):
        """Test inserting news into database."""
        count = db.insert_news(sample_news)
        assert count == len(sample_news)
    
    def test_get_news_count(self, db, sample_news):
        """Test getting news count."""
        db.insert_news(sample_news)
        count = db.get_news_count()
        assert count == len(sample_news)
    
    def test_insert_news_duplicates(self, db, sample_news):
        """Test that duplicate URLs are not inserted."""
        # Insert first time
        db.insert_news(sample_news)
        initial_count = db.get_news_count()
        
        # Insert same news again
        inserted = db.insert_news(sample_news)
        final_count = db.get_news_count()
        
        # Nothing should be inserted on second attempt
        assert inserted == 0
        assert final_count == initial_count
    
    def test_get_news_by_source(self, db, sample_news):
        """Test querying news by source."""
        db.insert_news(sample_news)
        result = db.get_news_by_source("TestSource")
        
        assert len(result) > 0
        assert all(news['source'] == "TestSource" for news in result)
    
    def test_get_latest_news(self, db, sample_news):
        """Test getting latest news."""
        db.insert_news(sample_news)
        result = db.get_latest_news(limit=1)
        
        assert len(result) == 1
        assert result[0]['title'] == sample_news[0]['title']
    
    def test_insert_empty_news_list(self, db):
        """Test inserting empty list."""
        count = db.insert_news([])
        assert count == 0
    
    def test_query_nonexistent_source(self, db, sample_news):
        """Test querying non-existent source."""
        db.insert_news(sample_news)
        result = db.get_news_by_source("NonExistentSource")
        
        assert len(result) == 0

    def test_get_all_news_and_update_news_batch(self, db, sample_news):
        """Test retrieving all news and updating enrichment fields."""
        # Insert news with enrichment fields
        enriched_news = []
        for news in sample_news:
            enriched_news.append({
                **news,
                "is_relevant": True,
                "segments": ["bancos"],
                "tickers": ["ITUB4"]
            })

        db.insert_news(enriched_news)

        all_news = db.get_all_news()
        assert len(all_news) == len(sample_news)

        updated = db.update_news_batch([
            {
                "url": sample_news[0]["link"],
                "is_relevant": False,
                "sentiment": "neutro",
                "confidence": 0.0,
                "segments": [],
                "tickers": []
            }
        ])

        assert updated == 1

        result = db.get_news_by_source("TestSource")
        assert result[0]["is_relevant"] == 0
        assert result[0]["sentiment"] == "neutro"
        assert result[0]["confidence"] == 0.0

    def test_delete_old_neutral_news_removes_stale_records(self, db):
        """Neutral news older than the cutoff should be removed."""
        old_neutral = {
            "title": "Old neutral news",
            "summary": "Old neutral content",
            "link": "https://example.com/old-neutral",
            "published_at": "2020-01-01T00:00:00",
            "source": "TestSource",
            "collected_at": "2020-01-01T01:00:00",
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": [],
        }
        db.insert_news([old_neutral])
        assert db.get_news_count() == 1

        deleted = db.delete_old_neutral_news(days=7)
        assert deleted == 1
        assert db.get_news_count() == 0

    def test_delete_old_neutral_news_preserves_recent_records(self, db):
        """Neutral news within the cutoff window should NOT be removed."""
        from datetime import datetime, timedelta, timezone
        recent_published = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%dT%H:%M:%S")

        recent_neutral = {
            "title": "Recent neutral news",
            "summary": "Recent neutral content",
            "link": "https://example.com/recent-neutral",
            "published_at": recent_published,
            "source": "TestSource",
            "collected_at": recent_published,
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": [],
        }
        db.insert_news([recent_neutral])
        assert db.get_news_count() == 1

        deleted = db.delete_old_neutral_news(days=7)
        assert deleted == 0
        assert db.get_news_count() == 1

    def test_delete_old_neutral_news_preserves_non_neutral_records(self, db):
        """Old but non-neutral (positive/negative) news should NOT be removed."""
        old_positive = {
            "title": "Old positive news",
            "summary": "Old positive content",
            "link": "https://example.com/old-positive",
            "published_at": "2020-01-01T00:00:00",
            "source": "TestSource",
            "collected_at": "2020-01-01T01:00:00",
            "sentiment": "positivo",
            "confidence": 0.9,
            "segments": [],
            "tickers": [],
        }
        db.insert_news([old_positive])
        assert db.get_news_count() == 1

        deleted = db.delete_old_neutral_news(days=7)
        assert deleted == 0
        assert db.get_news_count() == 1

    def test_get_latest_published_at_by_source_returns_max(self, db, sample_news):
        """Should return the most recent published_at for a source."""
        older = {
            "title": "Older article",
            "summary": "Older content",
            "link": "https://example.com/older",
            "published_at": "2026-03-20T08:00:00",
            "source": "TestSource",
            "collected_at": "2026-03-20T09:00:00",
        }
        newer = {
            "title": "Newer article",
            "summary": "Newer content",
            "link": "https://example.com/newer",
            "published_at": "2026-03-26T10:00:00",
            "source": "TestSource",
            "collected_at": "2026-03-26T11:00:00",
        }
        db.insert_news([older, newer])

        result = db.get_latest_published_at_by_source("TestSource")
        assert result == "2026-03-26T10:00:00"

    def test_get_latest_published_at_by_source_returns_none_when_empty(self, db):
        """Should return None when no records exist for the source."""
        result = db.get_latest_published_at_by_source("NonExistentSource")
        assert result is None

    def test_get_latest_published_at_by_source_isolates_sources(self, db):
        """Checkpoint for one source should not be affected by another source's data."""
        db.insert_news([
            {
                "title": "Source A article",
                "summary": "content",
                "link": "https://example.com/a",
                "published_at": "2026-04-01T10:00:00",
                "source": "SourceA",
                "collected_at": "2026-04-01T11:00:00",
            }
        ])

        # SourceB has no records
        result_b = db.get_latest_published_at_by_source("SourceB")
        assert result_b is None

        # SourceA checkpoint is intact
        result_a = db.get_latest_published_at_by_source("SourceA")
        assert result_a == "2026-04-01T10:00:00"
