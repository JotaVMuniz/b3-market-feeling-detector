"""
Raw data storage module for saving news as JSON files.
"""

import json
import logging
from typing import List, Dict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def get_raw_data_path(data_dir: str = "data/raw") -> Path:
    """
    Get the path to the raw data directory.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Path object pointing to raw data directory
    """
    raw_path = Path(data_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    return raw_path


def get_raw_filename(data_dir: str = "data/raw") -> str:
    """
    Generate a raw data filename based on current date.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Full path to the raw data file
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    raw_path = get_raw_data_path(data_dir)
    return str(raw_path / f"news_{today}.json")


def load_existing_news(filepath: str) -> List[Dict]:
    """
    Load existing news from a JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of news entries or empty list if file doesn't exist
    """
    try:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Loaded {len(data)} existing news from {filepath}")
                return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Error loading existing news from {filepath}: {str(e)}")
    
    return []


def save_raw_news(news_list: List[Dict], data_dir: str = "data/raw") -> str:
    """
    Save raw news data to JSON file.
    
    Appends to existing file if it exists (same day), otherwise creates new file.
    
    Args:
        news_list: List of news entries to save
        data_dir: Path to the data directory
        
    Returns:
        Path to the saved file
    """
    if not news_list:
        logger.warning("No news to save")
        return ""
    
    filepath = get_raw_filename(data_dir)
    
    try:
        # Load existing news if file exists
        existing_news = load_existing_news(filepath)
        
        # Merge with new news (avoid duplicates by URL)
        existing_urls = {news.get("link") for news in existing_news}
        new_unique_news = [
            news for news in news_list
            if news.get("link") not in existing_urls
        ]
        
        # Combine all news
        all_news = existing_news + new_unique_news
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_news, f, ensure_ascii=False, indent=2)
        
        logger.info(
            f"Saved {len(new_unique_news)} new news entries to {filepath} "
            f"(total: {len(all_news)})"
        )
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving raw news to {filepath}: {str(e)}")
        raise


def load_raw_news(filepath: str) -> List[Dict]:
    """
    Load raw news data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of news entries
    """
    return load_existing_news(filepath)
