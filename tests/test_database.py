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
