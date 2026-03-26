"""
Main orchestration module for the financial news ingestion pipeline.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

# Import pipeline modules
from src.ingestion.sources import get_sources
from src.ingestion.fetch_news import fetch_all_news, deduplicate_news
from src.processing.clean_news import clean_news_batch, validate_news_entry
from src.storage.save_raw import save_raw_news
from src.storage.database import NewsDatabase
from src.nlp.sentiment import classify_batch


def run_pipeline() -> None:
    """
    Execute the complete news ingestion pipeline.
    
    Pipeline steps:
    1. Fetch news from all configured RSS sources
    2. Deduplicate news based on URL
    3. Clean and process raw data
    4. Classify sentiment using LLM
    5. Save raw data to JSON files
    6. Insert cleaned data into SQLite database
    """
    logger.info("=" * 60)
    logger.info("Starting news ingestion pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Fetch news from all sources
        logger.info("\n[1/5] Fetching news from RSS sources...")
        sources = get_sources()
        logger.info(f"Configured sources: {len(sources)}")
        
        raw_news = fetch_all_news(sources)
        logger.info(f"Total news fetched: {len(raw_news)}")
        
        if not raw_news:
            logger.warning("No news fetched. Exiting pipeline.")
            return
        
        # Step 2: Deduplicate
        logger.info("\n[2/5] Deduplicating news articles...")
        deduplicated_news = deduplicate_news(raw_news)
        logger.info(f"News after deduplication: {len(deduplicated_news)}")
        
        # Step 3: Clean and process
        logger.info("\n[3/5] Cleaning and processing news data...")
        cleaned_news = clean_news_batch(deduplicated_news)
        
        # Validate entries
        valid_news = [
            news for news in cleaned_news
            if validate_news_entry(news)
        ]
        logger.info(f"Valid news entries: {len(valid_news)}")
        
        if not valid_news:
            logger.warning("No valid news entries after cleaning. Exiting pipeline.")
            return
        
        # Step 3.5: Classify sentiment
        logger.info("\n[3.5/6] Classifying sentiment for news articles...")
        
        # Prepare texts for classification (combine title and content)
        texts_to_classify = [
            f"{news.get('title', '')} {news.get('summary', '')}".strip()
            for news in valid_news
        ]
        
        # Classify in batch
        sentiment_results = classify_batch(texts_to_classify)
        
        # Add sentiment results to news entries
        for news, sentiment in zip(valid_news, sentiment_results):
            news['sentiment'] = sentiment['sentiment']
            news['confidence'] = sentiment['confidence']
        
        # Count results
        successful_classifications = sum(1 for s in sentiment_results if s['confidence'] > 0.0)
        failed_classifications = len(sentiment_results) - successful_classifications
        
        logger.info(f"Sentiment classification complete:")
        logger.info(f"  - Total processed: {len(sentiment_results)}")
        logger.info(f"  - Successful: {successful_classifications}")
        logger.info(f"  - Failed: {failed_classifications}")
        
        # Step 4: Save raw data
        logger.info("\n[4/6] Saving raw data to JSON files...")
        raw_file = save_raw_news(valid_news)
        logger.info(f"Raw data saved to: {raw_file}")
        
        # Step 5: Insert into database
        logger.info("\n[5/6] Inserting cleaned data into SQLite database...")
        db = NewsDatabase()
        
        # Get initial count
        initial_count = db.get_news_count()
        
        # Insert news
        inserted_count = db.insert_news(valid_news)
        
        # Get final count
        final_count = db.get_news_count()
        
        logger.info(f"Database stats: {initial_count} -> {final_count} records")
        logger.info(f"Inserted in this run: {inserted_count}")
        
        # Pipeline summary
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline execution summary:")
        logger.info(f"  - Total fetched: {len(raw_news)}")
        logger.info(f"  - After deduplication: {len(deduplicated_news)}")
        logger.info(f"  - Valid entries: {len(valid_news)}")
        logger.info(f"  - Sentiment classifications: {successful_classifications} successful, {failed_classifications} failed")
        logger.info(f"  - Successfully inserted: {inserted_count}")
        logger.info(f"  - Database total records: {final_count}")
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
