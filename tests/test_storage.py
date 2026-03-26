"""
Tests for the storage module.
"""

import pytest
import json
from pathlib import Path
from src.storage.save_raw import (
    load_existing_news,
    save_raw_news,
    load_raw_news,
)


class TestLoadExistingNews:
    """Tests for loading existing news files."""
    
    def test_load_existing_news_file_not_exists(self, temp_data_dir):
        """Test loading from non-existent file."""
        filepath = str(Path(temp_data_dir) / "nonexistent.json")
        result = load_existing_news(filepath)
        assert result == []
    
    def test_load_existing_news_valid_file(self, temp_data_dir, sample_news):
        """Test loading from valid JSON file."""
        filepath = str(Path(temp_data_dir) / "test.json")
        
        # Create test file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_news, f)
        
        result = load_existing_news(filepath)
        assert len(result) == len(sample_news)
        assert result[0]["title"] == sample_news[0]["title"]


class TestSaveRawNews:
    """Tests for saving raw news to JSON files."""
    
    def test_save_raw_news_empty_list(self, temp_data_dir):
        """Test saving empty news list."""
        result = save_raw_news([], data_dir=temp_data_dir)
        assert result == ""
    
    def test_save_raw_news_creates_file(self, temp_data_dir, sample_news):
        """Test that save_raw_news creates a file."""
        result = save_raw_news(sample_news, data_dir=temp_data_dir)
        assert result != ""
        assert Path(result).exists()
    
    def test_save_raw_news_formats_correctly(self, temp_data_dir, sample_news):
        """Test that saved data is correctly formatted."""
        filepath = save_raw_news(sample_news, data_dir=temp_data_dir)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) == len(sample_news)
    
    def test_save_raw_news_deduplicates(self, temp_data_dir, sample_news):
        """Test that duplicates are not added on subsequent saves."""
        # First save
        filepath = save_raw_news(sample_news, data_dir=temp_data_dir)
        
        # Second save with same data
        result = save_raw_news(sample_news, data_dir=temp_data_dir)
        
        with open(result, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Should still have same number of items (no duplicates)
        assert len(data) == len(sample_news)


class TestLoadRawNews:
    """Tests for loading raw news from JSON files."""
    
    def test_load_raw_news(self, temp_data_dir, sample_news):
        """Test loading raw news."""
        filepath = save_raw_news(sample_news, data_dir=temp_data_dir)
        result = load_raw_news(filepath)
        
        assert len(result) == len(sample_news)
        assert result[0]["title"] == sample_news[0]["title"]
