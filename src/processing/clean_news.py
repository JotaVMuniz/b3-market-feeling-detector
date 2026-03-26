"""
Data cleaning and processing module for news articles.
"""

import logging
import re
from typing import List, Dict, Optional
from html.parser import HTMLParser
from datetime import datetime
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


class HTMLStripper(HTMLParser):
    """Helper class to strip HTML tags from text."""
    
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    
    def handle_data(self, data):
        """Handle parsed data."""
        self.fed.append(data)
    
    def get_data(self) -> str:
        """Get cleaned data."""
        return ''.join(self.fed)


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Text possibly containing HTML tags
        
    Returns:
        Text without HTML tags
    """
    if not text:
        return ""
    
    try:
        stripper = HTMLStripper()
        stripper.feed(text)
        return stripper.get_data()
    except Exception as e:
        logger.warning(f"Error stripping HTML tags: {str(e)}")
        return text


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and special characters.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep common punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', '', text)
    
    return text.strip()


def standardize_date(date_str: Optional[str]) -> Optional[str]:
    """
    Parse and standardize date string to ISO format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        ISO formatted date string or None if parsing fails
    """
    if not date_str:
        return None
    
    try:
        # Try to parse with dateutil (handles many formats)
        parsed_date = date_parser.parse(date_str)
        return parsed_date.isoformat()
    except Exception as e:
        logger.warning(f"Could not parse date '{date_str}': {str(e)}")
        return date_str


def clean_news_entry(news: Dict) -> Dict:
    """
    Clean and process a single news entry.
    
    Args:
        news: Raw news entry dictionary
        
    Returns:
        Cleaned news entry dictionary
    """
    cleaned = {
        "title": normalize_text(news.get("title", "")),
        "summary": normalize_text(strip_html_tags(news.get("summary", ""))),
        "link": news.get("link", "").strip(),
        "published_at": standardize_date(news.get("published_at")),
        "source": news.get("source", "").strip(),
        "collected_at": news.get("collected_at", ""),
    }
    
    return cleaned


def clean_news_batch(news_list: List[Dict]) -> List[Dict]:
    """
    Clean and process a batch of news entries.
    
    Args:
        news_list: List of raw news entries
        
    Returns:
        List of cleaned news entries
    """
    cleaned_news = []
    
    for idx, news in enumerate(news_list):
        try:
            cleaned = clean_news_entry(news)
            cleaned_news.append(cleaned)
        except Exception as e:
            logger.error(f"Error cleaning news entry {idx}: {str(e)}")
            continue
    
    logger.info(f"Cleaned {len(cleaned_news)} out of {len(news_list)} news entries")
    
    return cleaned_news


def validate_news_entry(news: Dict) -> bool:
    """
    Validate a cleaned news entry.
    
    Args:
        news: News entry to validate
        
    Returns:
        True if entry is valid, False otherwise
    """
    required_fields = ["title", "link", "source"]
    
    for field in required_fields:
        if not news.get(field) or not str(news.get(field)).strip():
            logger.warning(f"Missing required field '{field}' in news entry")
            return False
    
    return True
