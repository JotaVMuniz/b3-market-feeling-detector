"""
Tests for the Twitter/X ingestion module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.ingestion.fetch_tweets import fetch_tweets, SOURCE_NAME, FINANCIAL_QUERY


class TestFetchTweets:
    """Tests for the fetch_tweets function."""

    def test_returns_empty_list_when_no_token(self, monkeypatch):
        """Should return [] and not raise when TWITTER_BEARER_TOKEN is absent."""
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
        result = fetch_tweets(bearer_token=None)
        assert result == []

    def test_returns_empty_list_when_api_returns_no_data(self, monkeypatch):
        """Should return [] when the API response contains no 'data' key."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response):
            result = fetch_tweets()
        assert result == []

    def test_normalises_tweets_to_news_schema(self, monkeypatch):
        """Should return dicts with the RSS news schema keys."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "123456",
                    "text": "Ibovespa sobe 1% nesta quarta-feira",
                    "created_at": "2026-04-07T10:00:00.000Z",
                }
            ]
        }

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response):
            result = fetch_tweets()

        assert len(result) == 1
        tweet = result[0]
        assert tweet["source"] == SOURCE_NAME
        assert tweet["title"] == "Ibovespa sobe 1% nesta quarta-feira"
        assert tweet["summary"] == "Ibovespa sobe 1% nesta quarta-feira"
        assert tweet["link"] == "https://twitter.com/i/web/status/123456"
        assert tweet["published_at"] == "2026-04-07T10:00:00.000Z"
        assert "collected_at" in tweet

    def test_title_truncated_to_280_chars(self, monkeypatch):
        """Tweet text longer than 280 chars should be truncated in 'title'."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")
        long_text = "a" * 300
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [{"id": "1", "text": long_text, "created_at": "2026-04-07T10:00:00Z"}]
        }

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response):
            result = fetch_tweets()

        assert len(result[0]["title"]) == 280
        assert result[0]["summary"] == long_text

    def test_handles_http_error_gracefully(self, monkeypatch):
        """Should return [] when the API returns an HTTP error."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")
        import requests as req

        mock_response = Mock()
        mock_response.status_code = 401
        http_error = req.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response):
            result = fetch_tweets()
        assert result == []

    def test_handles_connection_error_gracefully(self, monkeypatch):
        """Should return [] when a network error occurs."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")
        import requests as req

        with patch(
            "src.ingestion.fetch_tweets.requests.get",
            side_effect=req.exceptions.ConnectionError("Network unreachable"),
        ):
            result = fetch_tweets()
        assert result == []

    def test_max_results_clamped_to_100(self, monkeypatch):
        """max_results should be clamped to [10, 100]."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response) as mock_get:
            fetch_tweets(max_results=200)
            call_params = mock_get.call_args[1]["params"]
            assert call_params["max_results"] == 100

    def test_min_results_clamped_to_10(self, monkeypatch):
        """max_results below 10 should be raised to 10."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response) as mock_get:
            fetch_tweets(max_results=1)
            call_params = mock_get.call_args[1]["params"]
            assert call_params["max_results"] == 10

    def test_bearer_token_sent_in_header(self, monkeypatch):
        """The Authorization header must use the provided bearer token."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "my-secret-token")
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response) as mock_get:
            fetch_tweets()
            call_headers = mock_get.call_args[1]["headers"]
            assert call_headers["Authorization"] == "Bearer my-secret-token"

    def test_explicit_bearer_token_overrides_env(self, monkeypatch):
        """An explicit bearer_token argument should override the env variable."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "env-token")
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response) as mock_get:
            fetch_tweets(bearer_token="explicit-token")
            call_headers = mock_get.call_args[1]["headers"]
            assert call_headers["Authorization"] == "Bearer explicit-token"

    def test_fallback_published_at_when_created_at_absent(self, monkeypatch):
        """When 'created_at' is absent, 'published_at' should fall back to 'collected_at'."""
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [{"id": "99", "text": "Test tweet without created_at"}]
        }

        with patch("src.ingestion.fetch_tweets.requests.get", return_value=mock_response):
            result = fetch_tweets()

        assert len(result) == 1
        tweet = result[0]
        # published_at should equal collected_at as fallback
        assert tweet["published_at"] == tweet["collected_at"]

    def test_financial_query_contains_key_terms(self):
        """FINANCIAL_QUERY should include key Brazilian market terms."""
        for term in ("B3", "Ibovespa", "Petrobras", "lang:pt", "-is:retweet"):
            assert term in FINANCIAL_QUERY
