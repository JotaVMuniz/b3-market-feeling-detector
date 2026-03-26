"""
RSS feed sources configuration for financial news.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RSSSource:
    """Represents an RSS feed source."""
    
    name: str
    url: str
    
    def __str__(self) -> str:
        return f"{self.name} ({self.url})"


# Define RSS sources for Brazilian financial news
FINANCIAL_SOURCES: List[RSSSource] = [
    RSSSource(
        name="InfoMoney",
        url="https://www.infomoney.com.br/feed/"
    ),
    RSSSource(
        name="Valor Globo",
        url="https://valor.globo.com/rss/"
    ),
    RSSSource(
        name="Exame",
        url="https://exame.com/feed/"
    ),
]


def get_sources() -> List[RSSSource]:
    """
    Get all configured RSS sources.
    
    Returns:
        List of RSSSource objects
    """
    return FINANCIAL_SOURCES
