"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_news():
    """Sample news data for testing."""
    return [
        {
            "title": "Test News 1",
            "summary": "<p>Test summary with <b>HTML</b> tags</p>",
            "link": "https://example.com/news1",
            "published_at": "2026-03-26T10:00:00",
            "source": "TestSource",
            "collected_at": "2026-03-26T11:00:00"
        },
        {
            "title": "Test News 2",
            "summary": "Another test summary",
            "link": "https://example.com/news2",
            "published_at": "2026-03-25T10:00:00",
            "source": "TestSource",
            "collected_at": "2026-03-26T11:00:00"
        }
    ]


@pytest.fixture
def sample_raw_news():
    """Sample raw RSS news data."""
    return [
        {
            "title": "Raw News Title",
            "summary": "<p>Raw news with HTML</p>",
            "link": "https://example.com/raw",
            "published_at": None,
            "source": "RawSource",
            "collected_at": "2026-03-26T11:00:00"
        }
    ]
