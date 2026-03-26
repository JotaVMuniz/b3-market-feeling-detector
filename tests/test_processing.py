"""
Tests for the processing module.
"""

import pytest
from src.processing.clean_news import (
    strip_html_tags,
    normalize_text,
    standardize_date,
    clean_news_entry,
    clean_news_batch,
    validate_news_entry,
)


class TestHTMLStripper:
    """Tests for HTML tag stripping."""
    
    def test_strip_html_tags_basic(self):
        """Test basic HTML tag stripping."""
        html = "<p>Hello <b>World</b></p>"
        result = strip_html_tags(html)
        assert result == "Hello World"
    
    def test_strip_html_tags_empty(self):
        """Test with empty string."""
        result = strip_html_tags("")
        assert result == ""
    
    def test_strip_html_tags_none(self):
        """Test with None value - should return empty string."""
        result = strip_html_tags(None)
        assert result == ""
    
    def test_strip_html_tags_nested(self):
        """Test with nested HTML tags."""
        html = "<div><p>Nested <span>content</span></p></div>"
        result = strip_html_tags(html)
        assert result == "Nested content"


class TestNormalizeText:
    """Tests for text normalization."""
    
    def test_normalize_text_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    world   with   spaces"
        result = normalize_text(text)
        assert result == "Hello world with spaces"
    
    def test_normalize_text_special_chars(self):
        """Test special character removal."""
        text = "Hello@World#123"
        result = normalize_text(text)
        # Should keep alphanumeric and some punctuation
        assert "@" not in result
        assert "#" not in result
    
    def test_normalize_text_empty(self):
        """Test with empty string."""
        result = normalize_text("")
        assert result == ""
    
    def test_normalize_text_punctuation(self):
        """Test that common punctuation is preserved."""
        text = "Hello, world! How are you?"
        result = normalize_text(text)
        assert "," in result
        assert "!" in result
        assert "?" in result


class TestStandardizeDate:
    """Tests for date standardization."""
    
    def test_standardize_date_iso_format(self):
        """Test ISO format date."""
        date_str = "2026-03-26T10:30:00"
        result = standardize_date(date_str)
        assert result is not None
        assert "2026-03-26" in result
    
    def test_standardize_date_none(self):
        """Test with None value."""
        result = standardize_date(None)
        assert result is None
    
    def test_standardize_date_empty(self):
        """Test with empty string."""
        result = standardize_date("")
        assert result is None


class TestCleanNewsEntry:
    """Tests for single news entry cleaning."""
    
    def test_clean_news_entry_basic(self, sample_news):
        """Test cleaning a basic news entry."""
        news = sample_news[0]
        result = clean_news_entry(news)
        
        assert result["title"] is not None
        assert result["link"] == news["link"]
        assert result["source"] == news["source"]
        assert "<" not in result["summary"]
        assert ">" not in result["summary"]
    
    def test_clean_news_entry_missing_fields(self):
        """Test with missing optional fields."""
        news = {
            "title": "Test",
            "link": "https://example.com",
            "source": "Test"
        }
        result = clean_news_entry(news)
        
        assert result["title"] == "Test"
        assert result["summary"] == ""


class TestCleanNewsBatch:
    """Tests for batch news cleaning."""
    
    def test_clean_news_batch(self, sample_news):
        """Test cleaning a batch of news."""
        result = clean_news_batch(sample_news)
        
        assert len(result) == len(sample_news)
        assert all(isinstance(news, dict) for news in result)


class TestValidateNewsEntry:
    """Tests for news entry validation."""
    
    def test_validate_news_entry_valid(self, sample_news):
        """Test validation with valid entry."""
        news = clean_news_entry(sample_news[0])
        result = validate_news_entry(news)
        assert result is True
    
    def test_validate_news_entry_missing_title(self):
        """Test validation with missing title."""
        news = {
            "title": "",
            "link": "https://example.com",
            "source": "Test"
        }
        result = validate_news_entry(news)
        assert result is False
    
    def test_validate_news_entry_missing_link(self):
        """Test validation with missing link."""
        news = {
            "title": "Test",
            "link": "",
            "source": "Test"
        }
        result = validate_news_entry(news)
        assert result is False
    
    def test_validate_news_entry_missing_source(self):
        """Test validation with missing source."""
        news = {
            "title": "Test",
            "link": "https://example.com",
            "source": ""
        }
        result = validate_news_entry(news)
        assert result is False
