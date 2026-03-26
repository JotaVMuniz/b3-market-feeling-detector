"""
Tests for the NLP sentiment analysis module.
"""

import pytest
from unittest.mock import Mock, patch
from src.nlp.sentiment import (
    classify_sentiment,
    classify_batch,
    clear_cache,
    SentimentClassifier,
    _sentiment_cache
)


class TestSentimentClassifier:
    """Test the SentimentClassifier class."""

    def test_init_with_api_key(self):
        """Test initialization with provided API key."""
        classifier = SentimentClassifier(api_key="test-key")
        assert classifier.api_key == "test-key"

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        classifier = SentimentClassifier()
        assert classifier.api_key == "env-key"

    def test_init_no_api_key(self, monkeypatch):
        """Test initialization without API key raises error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            SentimentClassifier()

    def test_truncate_text(self):
        """Test text truncation."""
        classifier = SentimentClassifier(api_key="test")
        long_text = "a" * 1001
        truncated = classifier._truncate_text(long_text)
        assert len(truncated) == 1003  # 1000 + "..."
        assert truncated.endswith("...")

    def test_parse_response_valid(self):
        """Test parsing valid JSON response."""
        classifier = SentimentClassifier(api_key="test")
        response = '{"sentiment": "positivo", "confidence": 0.85}'
        result = classifier._parse_response(response)
        assert result == {"sentiment": "positivo", "confidence": 0.85}

    def test_parse_response_invalid_sentiment(self):
        """Test parsing response with invalid sentiment."""
        classifier = SentimentClassifier(api_key="test")
        response = '{"sentiment": "invalid", "confidence": 0.5}'
        result = classifier._parse_response(response)
        assert result == {"sentiment": "neutro", "confidence": 0.0}

    def test_parse_response_invalid_confidence(self):
        """Test parsing response with invalid confidence."""
        classifier = SentimentClassifier(api_key="test")
        response = '{"sentiment": "positivo", "confidence": 1.5}'
        result = classifier._parse_response(response)
        assert result == {"sentiment": "neutro", "confidence": 0.0}

    def test_parse_response_malformed_json(self):
        """Test parsing malformed JSON response."""
        classifier = SentimentClassifier(api_key="test")
        response = 'not json'
        result = classifier._parse_response(response)
        assert result == {"sentiment": "neutro", "confidence": 0.0}


class TestClassifySentiment:
    """Test the classify_sentiment function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    @patch('src.nlp.sentiment.SentimentClassifier._classify_single')
    def test_classify_sentiment_empty_text(self, mock_classify):
        """Test classification of empty text."""
        result = classify_sentiment("")
        assert result == {"sentiment": "neutro", "confidence": 0.0}
        mock_classify.assert_not_called()

    @patch('src.nlp.sentiment.SentimentClassifier._classify_single')
    def test_classify_sentiment_caching(self, mock_classify, monkeypatch):
        """Test that results are cached."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_classify.return_value = {"sentiment": "positivo", "confidence": 0.8}

        # First call
        result1 = classify_sentiment("test text")
        assert result1 == {"sentiment": "positivo", "confidence": 0.8}
        assert mock_classify.call_count == 1

        # Second call with same text should use cache
        result2 = classify_sentiment("test text")
        assert result2 == {"sentiment": "positivo", "confidence": 0.8}
        assert mock_classify.call_count == 1  # Still 1, used cache

    def test_classify_sentiment_api_call(self, monkeypatch):
        """Test successful API call."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        # Clear any existing classifier
        if hasattr(classify_sentiment, '_classifier'):
            delattr(classify_sentiment, '_classifier')
        
        with patch('src.nlp.sentiment.SentimentClassifier') as mock_classifier_class:
            mock_classifier = Mock()
            mock_classifier._classify_single.return_value = {"sentiment": "negativo", "confidence": 0.6}
            mock_classifier_class.return_value = mock_classifier

            result = classify_sentiment("market crash news")
            assert result == {"sentiment": "negativo", "confidence": 0.6}
            mock_classifier._classify_single.assert_called_once()


class TestClassifyBatch:
    """Test the classify_batch function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    @patch('src.nlp.sentiment.classify_sentiment')
    def test_classify_batch(self, mock_classify):
        """Test batch classification."""
        mock_classify.side_effect = [
            {"sentiment": "positivo", "confidence": 0.9},
            {"sentiment": "negativo", "confidence": 0.7},
            {"sentiment": "neutro", "confidence": 0.5}
        ]

        texts = ["good news", "bad news", "neutral news"]
        results = classify_batch(texts)

        assert len(results) == 3
        assert results[0] == {"sentiment": "positivo", "confidence": 0.9}
        assert results[1] == {"sentiment": "negativo", "confidence": 0.7}
        assert results[2] == {"sentiment": "neutro", "confidence": 0.5}
        assert mock_classify.call_count == 3


class TestCache:
    """Test cache functionality."""

    def test_clear_cache(self):
        """Test clearing the cache."""
        _sentiment_cache["test"] = {"sentiment": "positivo", "confidence": 1.0}
        assert len(_sentiment_cache) == 1

        clear_cache()
        assert len(_sentiment_cache) == 0