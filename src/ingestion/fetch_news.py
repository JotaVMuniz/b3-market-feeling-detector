"""
News ingestion module for fetching RSS feeds.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def setup_session(retries: int = 3, timeout: int = 10) -> requests.Session:
    """
    Create a requests session with retry logic.
    
    Args:
        retries: Number of retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        Configured requests.Session object
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def fetch_feed(source_url: str, source_name: str, session: Optional[requests.Session] = None) -> List[Dict]:
    """
    Fetch and parse a single RSS feed.
    
    Args:
        source_url: URL of the RSS feed
        source_name: Name of the news source
        session: Optional requests session for HTTP calls
        
    Returns:
        List of parsed news entries as dictionaries
    """
    news_entries = []
    
    try:
        logger.info(f"Fetching feed from {source_name}: {source_url}")
        
        # Use feedparser to fetch and parse the RSS feed
        feed = feedparser.parse(source_url)
        
        if feed.bozo:
            logger.warning(f"Feed parsing warning for {source_name}: {feed.bozo_exception}")
        
        if not feed.entries:
            logger.warning(f"No entries found in feed from {source_name}")
            return news_entries
        
        # Process each entry in the feed
        for entry in feed.entries:
            try:
                # Extract published date
                published_at = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_at = datetime(*entry.published_parsed[:6]).isoformat()
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published_at = datetime(*entry.updated_parsed[:6]).isoformat()
                
                # Extract summary/content
                summary = ""
                if hasattr(entry, 'summary'):
                    summary = entry.summary
                elif hasattr(entry, 'description'):
                    summary = entry.description
                
                # Build news item
                news_item = {
                    "title": entry.get("title", ""),
                    "summary": summary,
                    "link": entry.get("link", ""),
                    "published_at": published_at,
                    "source": source_name,
                    "collected_at": datetime.utcnow().isoformat()
                }
                
                # Only add if title and link are present
                if news_item["title"] and news_item["link"]:
                    news_entries.append(news_item)
                    
            except Exception as e:
                logger.error(f"Error parsing entry from {source_name}: {str(e)}")
                continue
        
        logger.info(f"Successfully fetched {len(news_entries)} entries from {source_name}")
        
    except Exception as e:
        logger.error(f"Error fetching feed from {source_name} ({source_url}): {str(e)}")
    
    return news_entries


def fetch_all_news(sources: List) -> List[Dict]:
    """
    Fetch news from all configured sources.
    
    Args:
        sources: List of RSSSource objects
        
    Returns:
        List of all collected news entries
    """
    all_news = []
    session = setup_session()
    
    for source in sources:
        news = fetch_feed(source.url, source.name, session)
        all_news.extend(news)
    
    session.close()
    logger.info(f"Total news collected: {len(all_news)}")
    
    return all_news


def deduplicate_news(news_list: List[Dict]) -> List[Dict]:
    """
    Remove duplicate news based on URL.
    
    Args:
        news_list: List of news entries
        
    Returns:
        Deduplicated list of news entries
    """
    seen_links = set()
    deduplicated = []
    
    for news in news_list:
        link = news.get("link", "")
        if link and link not in seen_links:
            seen_links.add(link)
            deduplicated.append(news)
        else:
            logger.debug(f"Duplicate news found: {news.get('title', 'Unknown')}")
    
    logger.info(f"Deduplicated {len(news_list) - len(deduplicated)} duplicate entries")
    
    return deduplicated
