"""
Tests for the NLP enrichment module.
"""

import json
from unittest.mock import Mock, patch

import pytest

from src.nlp.enrichment import (
    EnrichmentClassifier,
    clear_cache,
    enrich_batch,
    enrich_news,
    is_probably_financial,
    _enrichment_cache,
)


class TestFinancialFilter:
    def test_is_probably_financial_true(self):
        text = "Empresa anuncia lucro e alta das ações no setor bancário."
        assert is_probably_financial(text) is True

    def test_is_probably_financial_false(self):
        text = "Chuva forte atinge a cidade e causa transtornos no trânsito."
        assert is_probably_financial(text) is False


class TestEnrichmentClassifier:
    def test_parse_response_valid_relevant(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("src.nlp.enrichment.OpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            classifier = EnrichmentClassifier()

        response = json.dumps({
            "is_relevant": True,
            "sentiment": "positivo",
            "confidence": 0.9,
            "segments": ["bancos", "varejo"],
            "tickers": ["ITUB4", "VALE3"]
        })

        result = classifier._parse_response(response)

        assert result == {
            "is_relevant": True,
            "sentiment": "positivo",
            "confidence": 0.9,
            "segments": ["bancos", "varejo"],
            "tickers": ["ITUB4", "VALE3"]
        }

    def test_parse_response_invalid_json_returns_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("src.nlp.enrichment.OpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            classifier = EnrichmentClassifier()

        response = "not json"
        result = classifier._parse_response(response)

        assert result == {
            "is_relevant": False,
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": []
        }

    def test_parse_response_invalid_segments_and_tickers_filtered(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("src.nlp.enrichment.OpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            classifier = EnrichmentClassifier()

        response = json.dumps({
            "is_relevant": True,
            "sentiment": "positivo",
            "confidence": 0.7,
            "segments": ["bancos", "esporte"],
            "tickers": ["ITUB4", "INVALID"]
        })

        result = classifier._parse_response(response)

        assert result == {
            "is_relevant": True,
            "sentiment": "positivo",
            "confidence": 0.7,
            "segments": ["bancos"],
            "tickers": ["ITUB4"]
        }


class TestEnrichNews:
    def setup_method(self):
        clear_cache()

    def test_enrich_news_skips_api_for_non_financial_text(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("src.nlp.enrichment.EnrichmentClassifier._enrich_single") as mock_enrich_single:
            result = enrich_news("Chuva e trânsito causam alagamento hoje.")

        assert result == {
            "is_relevant": False,
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": []
        }
        assert mock_enrich_single.call_count == 0

    def test_enrich_news_caches_results(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("src.nlp.enrichment.EnrichmentClassifier._enrich_single") as mock_enrich_single:
            mock_enrich_single.return_value = {
                "is_relevant": True,
                "sentiment": "positivo",
                "confidence": 0.8,
                "segments": ["bancos"],
                "tickers": ["ITUB4"]
            }

            text = "Empresa anuncia aquisição e expectativas de lucro."
            result1 = enrich_news(text)
            result2 = enrich_news(text)

        assert result1 == result2
        assert mock_enrich_single.call_count == 1
        assert text.strip()[:1000] in _enrichment_cache

    def test_enrich_news_fallback_on_exception(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("src.nlp.enrichment.EnrichmentClassifier._enrich_single", side_effect=Exception("API error")):
            result = enrich_news("Empresa anuncia aquisição e expectativas de lucro.")

        assert result == {
            "is_relevant": False,
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": []
        }


class TestEnrichBatch:
    def setup_method(self):
        clear_cache()

    def test_enrich_batch_returns_results_for_all_texts(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("src.nlp.enrichment.enrich_news") as mock_enrich_news:
            mock_enrich_news.side_effect = [
                {"is_relevant": True, "sentiment": "positivo", "confidence": 0.8, "segments": ["bancos"], "tickers": ["ITUB4"]},
                {"is_relevant": False, "sentiment": "neutro", "confidence": 0.0, "segments": [], "tickers": []}
            ]

            texts = [
                "Empresa anuncia lucro.",
                "Chuva atinge a cidade."
            ]
            results = enrich_batch(texts)

        assert len(results) == 2
        assert results[0]["is_relevant"] is True
        assert results[1]["is_relevant"] is False
        assert mock_enrich_news.call_count == 2

    def test_clear_cache(self):
        _enrichment_cache["test"] = {"is_relevant": False, "sentiment": "neutro", "confidence": 0.0, "segments": [], "tickers": []}
        assert len(_enrichment_cache) == 1

        clear_cache()
        assert _enrichment_cache == {}
