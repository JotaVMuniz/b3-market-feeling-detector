"""
NLP module for structured information extraction from financial news using OpenAI API.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
import openai
from openai import OpenAI

# Carrega variáveis do arquivo .env automaticamente
load_dotenv()

logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 1000
MODEL_NAME = "gpt-4o-mini"
MAX_RETRIES = 2
VALID_SENTIMENTS = ["positivo", "negativo", "neutro"]
ALLOWED_SEGMENTS = ["petróleo", "mineração", "bancos", "varejo", "energia", "tecnologia", "agronegócio", "logística"]
TICKER_PATTERN = re.compile(r'^[A-Z]{4}\d$')

# Financial keywords for pre-filtering
FINANCIAL_KEYWORDS = {
    'empresa', 'empresas', 'lucro', 'lucros', 'ações', 'ação', 'bolsa', 'b3', 'bovespa',
    'petrobras', 'vale', 'itau', 'bradesco', 'banco', 'bancos', 'mercado', 'mercados',
    'investimento', 'investimentos', 'economia', 'econômico', 'econômica', 'setor', 'setores',
    'indústria', 'indústrias', 'commodities', 'commodity', 'preço', 'preços', 'cotação',
    'cotações', 'dividendos', 'dividendo', 'ipo', 'fusão', 'aquisição', 'm&a', 'fusões',
    'aquisições', 'regulação', 'regulatório', 'regulamentação', 'inflação', 'juros',
    'taxa', 'taxas', 'selic', 'dólar', 'dolar', 'câmbio', 'exportação', 'exportações',
    'importação', 'importações', 'balança', 'comercial', 'pib', 'produto interno bruto'
}


def is_probably_financial(text: str) -> bool:
    """
    Lightweight rule-based filter to check if text is likely financial.
    
    Args:
        text: The news text to check.
        
    Returns:
        True if text contains financial keywords, False otherwise.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in FINANCIAL_KEYWORDS)


# Global cache for enrichment results
_enrichment_cache: Dict[str, Dict] = {}


class EnrichmentClassifier:
    """Handles structured information extraction using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the enrichment classifier.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        try:
            self.client = OpenAI(api_key=self.api_key)
        except TypeError:
            # Fallback for compatibility issues
            self.client = OpenAI(api_key=self.api_key, default_headers={})

    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum length."""
        if len(text) > MAX_TEXT_LENGTH:
            return text[:MAX_TEXT_LENGTH] + "..."
        return text

    def _build_prompt(self, text: str) -> str:
        """Build the enrichment prompt."""
        return f"""You are a financial analyst specialized in the Brazilian stock market (B3).

Analyze the following news.

Step 1: Determine if the news is relevant to financial markets, publicly traded companies, or economic sectors.

Relevant examples:

* Company earnings
* Mergers and acquisitions
* Regulation affecting industries
* Commodity prices
* Macroeconomic impacts on sectors

NOT relevant examples:

* Accidents
* Crime
* Weather
* General events unrelated to companies or markets

---

If the news is NOT relevant, respond ONLY with:

{{
"is_relevant": false,
"market_relevance": 0.0,
"sentiment": "neutro",
"confidence": 0.0,
"segments": [],
"tickers": []
}}

---

If the news IS relevant, respond ONLY with:

{{
"is_relevant": true,
"market_relevance": number between 0 and 1,
"sentiment": "positivo | negativo | neutro",
"confidence": number between 0 and 1,
"segments": ["segment1", "segment2"],
"tickers": ["TICKER1", "TICKER2"]
}}

---

Rules:

* Do NOT hallucinate segments or tickers
* If unsure, return empty lists []
* Segments must be chosen only from this list:
  ["petróleo", "mineração", "bancos", "varejo", "energia", "tecnologia", "agronegócio", "logística"]
* Tickers must be valid Brazilian stock tickers (e.g., PETR4, VALE3)
* If no ticker is clearly mentioned, return []
* sentiment must be strictly one of: positivo, negativo, neutro
* confidence must be between 0 and 1
* market_relevance is a score from 0 to 1 indicating how relevant and impactful the news is for the Brazilian financial market (0 = not relevant at all, 1 = highly relevant, market-moving news)
* Output MUST be valid JSON only (no explanations)

News:
"{text}"
"""

    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse and validate the API response.

        Returns:
            Dict with enrichment data, or fallback on failure.
        """
        try:
            data = json.loads(response_text.strip())

            # Validate is_relevant
            is_relevant = data.get("is_relevant", False)
            if not isinstance(is_relevant, bool):
                raise ValueError(f"Invalid is_relevant: {is_relevant}")

            # Validate market_relevance
            market_relevance = data.get("market_relevance", 0.0)
            if not isinstance(market_relevance, (int, float)) or not (0.0 <= market_relevance <= 1.0):
                raise ValueError(f"Invalid market_relevance: {market_relevance}")

            # Validate sentiment
            sentiment = data.get("sentiment", "").lower()
            if sentiment not in VALID_SENTIMENTS:
                raise ValueError(f"Invalid sentiment: {sentiment}")

            # Validate confidence
            confidence = data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Invalid confidence: {confidence}")

            # Validate segments
            segments = data.get("segments", [])
            if not isinstance(segments, list) or not all(isinstance(s, str) for s in segments):
                raise ValueError(f"Invalid segments: {segments}")
            # Filter to allowed segments only
            segments = [s for s in segments if s in ALLOWED_SEGMENTS]

            # Validate tickers
            tickers = data.get("tickers", [])
            if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
                raise ValueError(f"Invalid tickers: {tickers}")
            # Filter to valid tickers only
            tickers = [t for t in tickers if TICKER_PATTERN.match(t)]

            # Anti-hallucination: if not relevant, force empty lists and zero relevance
            if not is_relevant:
                segments = []
                tickers = []
                market_relevance = 0.0

            return {
                "is_relevant": is_relevant,
                "market_relevance": float(market_relevance),
                "sentiment": sentiment,
                "confidence": float(confidence),
                "segments": segments,
                "tickers": tickers
            }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse response: {response_text}. Error: {str(e)}")
            return {
                "is_relevant": False,
                "market_relevance": 0.0,
                "sentiment": "neutro",
                "confidence": 0.0,
                "segments": [],
                "tickers": []
            }

    def _enrich_single(self, text: str) -> Dict:
        """
        Enrich a single text with structured information.

        Returns:
            Dict with is_relevant, sentiment, confidence, segments, tickers.
        """
        truncated_text = self._truncate_text(text)
        prompt = self._build_prompt(truncated_text)

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0  # Deterministic responses
                )

                response_text = response.choices[0].message.content
                result = self._parse_response(response_text)

                logger.debug(f"Enrichment successful for text: {truncated_text[:50]}...")
                return result

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)}")
                if attempt == MAX_RETRIES:
                    logger.error(f"All retries failed for text: {truncated_text[:50]}...", exc_info=True)
                    return {
                        "is_relevant": False,
                        "market_relevance": 0.0,
                        "sentiment": "neutro",
                        "confidence": 0.0,
                        "segments": [],
                        "tickers": []
                    }

        return {
            "is_relevant": False,
            "market_relevance": 0.0,
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": []
        }


def enrich_news(text: str) -> Dict:
    """
    Extract structured market intelligence from financial news text.

    Uses caching to avoid duplicate API calls and pre-filtering for cost optimization.

    Args:
        text: The news text to enrich.

    Returns:
        Dict with keys:
        - is_relevant: bool (whether news is financially relevant)
        - market_relevance: float between 0 and 1 (relevance score for the Brazilian financial market)
        - sentiment: "positivo" | "negativo" | "neutro"
        - confidence: float between 0 and 1
        - segments: list of str (market segments)
        - tickers: list of str (Brazilian stock tickers)
    """
    if not text or not text.strip():
        return {
            "is_relevant": False,
            "market_relevance": 0.0,
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": []
        }

    # Check cache first
    cache_key = text.strip()[:MAX_TEXT_LENGTH]  # Use truncated text as key
    if cache_key in _enrichment_cache:
        logger.debug("Using cached enrichment result")
        return _enrichment_cache[cache_key]

    # Pre-filter: skip API call if not probably financial
    if not is_probably_financial(text):
        logger.debug("Text filtered out as non-financial, skipping API call")
        result = {
            "is_relevant": False,
            "market_relevance": 0.0,
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": []
        }
        # Cache the result
        _enrichment_cache[cache_key] = result
        return result

    # Initialize classifier if needed
    if not hasattr(enrich_news, '_classifier'):
        enrich_news._classifier = EnrichmentClassifier()

    try:
        result = enrich_news._classifier._enrich_single(text)
    except Exception as e:
        logger.error("News enrichment failed, applying fallback", exc_info=True)
        result = {
            "is_relevant": False,
            "market_relevance": 0.0,
            "sentiment": "neutro",
            "confidence": 0.0,
            "segments": [],
            "tickers": []
        }

    # Cache the result
    _enrichment_cache[cache_key] = result

    return result


def enrich_batch(texts: List[str]) -> List[Dict]:
    """
    Enrich a batch of texts with structured information.

    Args:
        texts: List of news texts to enrich.

    Returns:
        List of enrichment dictionaries in the same order as input texts.
    """
    results = []
    skipped_count = 0
    api_calls = 0
    successful = 0
    failed = 0
    
    for text in texts:
        try:
            result = enrich_news(text)
            results.append(result)
            
            if not is_probably_financial(text):
                skipped_count += 1
            else:
                api_calls += 1
                if result.get('is_relevant', False) and result.get('confidence', 0) > 0:
                    successful += 1
                elif result.get('is_relevant', False):
                    failed += 1  # API called but invalid response
                    
        except Exception as e:
            logger.error("Batch enrichment failed for one text, applying fallback", exc_info=True)
            result = {
                "is_relevant": False,
                "market_relevance": 0.0,
                "sentiment": "neutro",
                "confidence": 0.0,
                "segments": [],
                "tickers": []
            }
            results.append(result)
            failed += 1
    
    logger.info(f"Enrichment batch complete: {len(texts)} total, {skipped_count} skipped (non-financial), {api_calls} API calls, {successful} successful, {failed} failed")
    return results


def clear_cache() -> None:
    """Clear the enrichment cache."""
    global _enrichment_cache
    _enrichment_cache.clear()
    logger.info("Enrichment cache cleared")