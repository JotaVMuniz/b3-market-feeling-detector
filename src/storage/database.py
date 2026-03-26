"""
Database module for storing news in SQLite.
"""

import sqlite3
import logging
from typing import List, Dict, Optional
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NewsDatabase:
    """SQLite database handler for news storage."""
    
    def __init__(self, db_path: str = "data/news.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self.init_database()
    
    def _ensure_db_directory(self) -> None:
        """Ensure the directory for the database exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            sqlite3.Connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            conn.close()
    
    def init_database(self) -> None:
        """Initialize the database schema."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create news table with unique constraint on URL
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS news (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        content TEXT,
                        source TEXT NOT NULL,
                        published_at TEXT,
                        url TEXT NOT NULL UNIQUE,
                        collected_at TEXT NOT NULL
                    )
                """)
                
                # Create index on source and published_at for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_source 
                    ON news(source)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_published_at 
                    ON news(published_at)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_collected_at 
                    ON news(collected_at)
                """)
                
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def insert_news(self, news_list: List[Dict]) -> int:
        """
        Insert news entries into the database.
        
        Skips entries with duplicate URLs (unique constraint).
        
        Args:
            news_list: List of news dictionaries
            
        Returns:
            Number of successfully inserted records
        """
        if not news_list:
            logger.warning("No news to insert")
            return 0
        
        inserted_count = 0
        skipped_count = 0
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for news in news_list:
                    try:
                        cursor.execute("""
                            INSERT INTO news 
                            (title, content, source, published_at, url, collected_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            news.get("title", ""),
                            news.get("summary", ""),
                            news.get("source", ""),
                            news.get("published_at"),
                            news.get("link", ""),
                            news.get("collected_at", "")
                        ))
                        inserted_count += 1
                    
                    except sqlite3.IntegrityError:
                        # URL already exists, skip
                        skipped_count += 1
                        logger.debug(f"Skipped duplicate: {news.get('link')}")
                    
                    except Exception as e:
                        logger.error(f"Error inserting news: {str(e)}")
                        continue
            
            logger.info(
                f"Database insert complete: {inserted_count} inserted, "
                f"{skipped_count} skipped (duplicates)"
            )
            return inserted_count
        
        except Exception as e:
            logger.error(f"Error inserting news batch: {str(e)}")
            return 0
    
    def get_news_count(self) -> int:
        """
        Get total number of news records in database.
        
        Returns:
            Count of news records
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM news")
                result = cursor.fetchone()
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting news count: {str(e)}")
            return 0
    
    def get_news_by_source(self, source: str, limit: int = 10) -> List[Dict]:
        """
        Get news entries by source.
        
        Args:
            source: Source name
            limit: Maximum number of records to return
            
        Returns:
            List of news records
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM news 
                    WHERE source = ? 
                    ORDER BY published_at DESC 
                    LIMIT ?
                """, (source, limit))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error querying news by source: {str(e)}")
            return []
    
    def get_latest_news(self, limit: int = 10) -> List[Dict]:
        """
        Get the latest news entries.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of news records ordered by published date
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM news 
                    ORDER BY published_at DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error querying latest news: {str(e)}")
            return []
