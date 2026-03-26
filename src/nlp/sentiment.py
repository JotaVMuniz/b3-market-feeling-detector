"""
NLP module for sentiment analysis of financial news using OpenAI API.
"""

import json
import logging
import os
from typing import Dict, List, Optional
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 1000
MODEL_NAME = "gpt-4o-mini"
MAX_RETRIES = 2
VALID_SENTIMENTS = ["positivo", "negativo", "neutro"]

# Global cache for sentiment results
_sentiment_cache: Dict[str, Dict] = {}


class SentimentClassifier:
    """Handles sentiment classification using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the sentiment classifier.

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
        """Build the classification prompt."""
        return f"""You are a financial analyst.

Classify the sentiment of the following news regarding its impact on the mentioned company or asset.

Respond ONLY in valid JSON format like this:
{{
"sentiment": "positivo | negativo | neutro",
"confidence": number between 0 and 1
}}

Rules:

* sentiment must be exactly one of: positivo, negativo, neutro
* confidence must be a float between 0 and 1
* do not include explanations
* do not include extra text

News:
"{text}"
"""

    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse and validate the API response.

        Returns:
            Dict with sentiment and confidence, or fallback on failure.
        """
        try:
            data = json.loads(response_text.strip())

            # Validate sentiment
            sentiment = data.get("sentiment", "").lower()
            if sentiment not in VALID_SENTIMENTS:
                raise ValueError(f"Invalid sentiment: {sentiment}")

            # Validate confidence
            confidence = data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Invalid confidence: {confidence}")

            return {
                "sentiment": sentiment,
                "confidence": float(confidence)
            }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse response: {response_text}. Error: {str(e)}")
            return {
                "sentiment": "neutro",
                "confidence": 0.0
            }

    def _classify_single(self, text: str) -> Dict:
        """
        Classify sentiment for a single text.

        Returns:
            Dict with sentiment and confidence.
        """
        truncated_text = self._truncate_text(text)
        prompt = self._build_prompt(truncated_text)

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.0  # Deterministic responses
                )

                response_text = response.choices[0].message.content
                result = self._parse_response(response_text)

                logger.debug(f"Classification successful for text: {truncated_text[:50]}...")
                return result

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)}")
                if attempt == MAX_RETRIES:
                    logger.error(f"All retries failed for text: {truncated_text[:50]}...")
                    return {
                        "sentiment": "neutro",
                        "confidence": 0.0
                    }

        return {
            "sentiment": "neutro",
            "confidence": 0.0
        }


def classify_sentiment(text: str) -> Dict:
    """
    Classify the sentiment of financial news text.

    Uses caching to avoid duplicate API calls.

    Args:
        text: The news text to classify.

    Returns:
        Dict with keys:
        - sentiment: "positivo" | "negativo" | "neutro"
        - confidence: float between 0 and 1
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutro",
            "confidence": 0.0
        }

    # Check cache first
    cache_key = text.strip()[:MAX_TEXT_LENGTH]  # Use truncated text as key
    if cache_key in _sentiment_cache:
        logger.debug("Using cached sentiment result")
        return _sentiment_cache[cache_key]

    # Initialize classifier if needed
    if not hasattr(classify_sentiment, '_classifier'):
        classify_sentiment._classifier = SentimentClassifier()

    result = classify_sentiment._classifier._classify_single(text)

    # Cache the result
    _sentiment_cache[cache_key] = result

    return result


def classify_batch(texts: List[str]) -> List[Dict]:
    """
    Classify sentiment for a batch of texts.

    Args:
        texts: List of news texts to classify.

    Returns:
        List of sentiment dictionaries in the same order as input texts.
    """
    results = []
    for text in texts:
        result = classify_sentiment(text)
        results.append(result)
    return results


def clear_cache() -> None:
    """Clear the sentiment cache."""
    global _sentiment_cache
    _sentiment_cache.clear()
    logger.info("Sentiment cache cleared")