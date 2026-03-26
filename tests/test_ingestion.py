"""
Tests for the ingestion module.
"""

import pytest
from src.ingestion.sources import get_sources, RSSSource
from src.ingestion.fetch_news import deduplicate_news


class TestRSSSource:
    """Tests for RSSSource dataclass."""
    
    def test_rss_source_creation(self):
        """Test creating an RSSSource."""
        source = RSSSource(name="Test", url="https://example.com/feed")
        assert source.name == "Test"
        assert source.url == "https://example.com/feed"
    
    def test_rss_source_str(self):
        """Test RSSSource string representation."""
        source = RSSSource(name="Test", url="https://example.com/feed")
        assert str(source) == "Test (https://example.com/feed)"


class TestGetSources:
    """Tests for getting configured sources."""
    
    def test_get_sources_returns_list(self):
        """Test that get_sources returns a list."""
        sources = get_sources()
        assert isinstance(sources, list)
    
    def test_get_sources_count(self):
        """Test that we have configured sources."""
        sources = get_sources()
        assert len(sources) > 0
    
    def test_get_sources_types(self):
        """Test that sources are RSSSource instances."""
        sources = get_sources()
        assert all(isinstance(source, RSSSource) for source in sources)
    
    def test_get_sources_have_urls(self):
        """Test that all sources have URLs."""
        sources = get_sources()
        assert all(source.url.startswith("http") for source in sources)


class TestDeduplicateNews:
    """Tests for news deduplication."""
    
    def test_deduplicate_news_removes_duplicates(self, sample_news):
        """Test that duplicates are removed."""
        # Add a duplicate
        duplicated = sample_news + [sample_news[0]]
        result = deduplicate_news(duplicated)
        
        assert len(result) == len(sample_news)
    
    def test_deduplicate_news_preserves_order(self, sample_news):
        """Test that deduplication preserves order."""
        result = deduplicate_news(sample_news)
        
        for i, news in enumerate(result):
            assert news["link"] == sample_news[i]["link"]
    
    def test_deduplicate_news_empty_list(self):
        """Test deduplication with empty list."""
        result = deduplicate_news([])
        assert result == []
    
    def test_deduplicate_news_no_duplicates(self, sample_news):
        """Test deduplication when there are no duplicates."""
        result = deduplicate_news(sample_news)
        assert len(result) == len(sample_news)
    
    def test_deduplicate_news_empty_link(self):
        """Test deduplication ignores entries with empty links."""
        news = [
            {"title": "Test", "link": "https://example.com/1"},
            {"title": "Test2", "link": ""},
            {"title": "Test3", "link": "https://example.com/2"}
        ]
        result = deduplicate_news(news)
        
        # Empty link should be ignored (not added to deduplicated list)
        assert len(result) == 2
        assert all(n["link"] for n in result)  # All remaining have non-empty links
