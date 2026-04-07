"""
Twitter/X API integration for fetching posts about the Brazilian financial market.

Uses the Twitter API v2 endpoint ``/2/tweets/search/recent`` which returns posts
from the last 7 days.  The free-tier plan allows up to 100 requests/month and
up to 100 results per request.

Required environment variable:
    TWITTER_BEARER_TOKEN: Bearer Token obtained from the X Developer Portal.
    See the README for setup instructions.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TWITTER_API_URL = "https://api.twitter.com/2/tweets/search/recent"

# Source label used when storing tweets in the news table
SOURCE_NAME = "X (Twitter)"

# Default search query — Brazilian financial market keywords in Portuguese.
# Retweets are excluded to avoid duplicate content.
FINANCIAL_QUERY = (
    "(B3 OR Ibovespa OR IBOV OR \"bolsa de valores\" OR Petrobras "
    "OR PETR4 OR VALE3 OR \"taxa Selic\" OR dólar OR \"mercado financeiro\" "
    "OR ações OR \"mercado de ações\" OR \"renda variável\" OR FII "
    "OR Bradesco OR Itaú OR XP OR \"BTG Pactual\") lang:pt -is:retweet"
)


def fetch_tweets(
    query: str = FINANCIAL_QUERY,
    max_results: int = 100,
    bearer_token: Optional[str] = None,
    start_time: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch recent tweets about the Brazilian financial market from the X API v2.

    When ``TWITTER_BEARER_TOKEN`` is absent the function returns an empty list
    and logs a warning so the rest of the pipeline can continue uninterrupted.

    Args:
        query: Full search query string sent to the API.  Defaults to
               :data:`FINANCIAL_QUERY` (Brazilian financial market keywords
               in Portuguese, retweets excluded).
        max_results: Number of results to request.  Clamped to the allowed
                     range 10–100 for the free-tier plan.
        bearer_token: Bearer token.  If *None*, reads from the
                      ``TWITTER_BEARER_TOKEN`` environment variable.
        start_time: RFC 3339 / ISO 8601 datetime string.  When provided, only
                    tweets created **after** this timestamp are returned.  Use
                    the ``published_at`` of the most recently stored tweet as a
                    checkpoint to avoid re-fetching already-ingested posts.

    Returns:
        List of tweet dictionaries normalised to the same schema used by the
        RSS news entries (keys: ``title``, ``summary``, ``link``,
        ``published_at``, ``source``, ``collected_at``).
    """
    token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        logger.warning(
            "TWITTER_BEARER_TOKEN not configured — skipping Twitter/X ingestion. "
            "See README for setup instructions."
        )
        return []

    headers = {"Authorization": f"Bearer {token}"}
    params: Dict = {
        "query": query,
        "max_results": max(10, min(int(max_results), 100)),
        "tweet.fields": "created_at,author_id,text",
    }
    if start_time:
        params["start_time"] = start_time
        logger.info("Tweet checkpoint: fetching posts after %s", start_time)

    try:
        response = requests.get(
            TWITTER_API_URL, headers=headers, params=params, timeout=15
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "N/A"
        logger.error("Twitter API HTTP error %s: %s", status, exc)
        return []
    except Exception as exc:
        logger.error("Twitter API request failed: %s", exc)
        return []

    tweets = data.get("data") or []
    if not tweets:
        logger.info("No tweets returned for the financial query.")
        return []

    collected_at = datetime.now(timezone.utc).isoformat()
    result: List[Dict] = []
    for tweet in tweets:
        tweet_id = tweet.get("id", "")
        text = tweet.get("text", "")
        created_at = tweet.get("created_at") or collected_at

        result.append(
            {
                "title": text[:280],
                "summary": text,
                "link": f"https://twitter.com/i/web/status/{tweet_id}",
                "published_at": created_at,
                "source": SOURCE_NAME,
                "collected_at": collected_at,
            }
        )

    logger.info("Fetched %d tweets from X API.", len(result))
    return result
