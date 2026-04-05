"""
Main orchestration module for the financial news ingestion pipeline.
"""

import argparse
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
from src.nlp.enrichment import enrich_batch, is_probably_financial


def run_pipeline(reprocess_existing: bool = False) -> None:
    """
    Execute the complete news ingestion pipeline.

    Pipeline steps:
    1. Fetch news from all configured RSS sources
    2. Deduplicate news based on URL
    3. Clean and process raw data
    4. Enrich news with structured information using LLM
    5. Save raw data to JSON files
    6. Insert cleaned data into SQLite database
    7. Optionally reprocess all existing news in the database
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
            logger.warning("No valid news entries after cleaning.")
            if not reprocess_existing:
                logger.warning("Exiting pipeline because reprocess_existing is False.")
                return
            logger.info("Continuing pipeline because reprocess_existing is True.")
        
        # Step 3.5: Enrich news with structured information
        logger.info("\n[3.5/6] Enriching news articles with structured information...")
        
        # Prepare texts for enrichment (combine title and content)
        texts_to_enrich = [
            f"{news.get('title', '')} {news.get('summary', '')}".strip()
            for news in valid_news
        ]
        
        # Enrich in batch
        enrichment_results = enrich_batch(texts_to_enrich)
        
        # Add enrichment results to news entries
        for news, enrichment in zip(valid_news, enrichment_results):
            news['is_relevant'] = enrichment['is_relevant']
            news['sentiment'] = enrichment['sentiment']
            news['confidence'] = enrichment['confidence']
            news['segments'] = enrichment['segments']
            news['tickers'] = enrichment['tickers']
        
        # Count results
        successful_enrichments = sum(1 for e in enrichment_results if e['is_relevant'] and e['confidence'] > 0.0)
        failed_enrichments = len(enrichment_results) - successful_enrichments
        skipped_non_financial = sum(1 for text in texts_to_enrich if not is_probably_financial(text))
        relevant_articles = sum(1 for e in enrichment_results if e['is_relevant'])
        invalid_responses = sum(1 for e in enrichment_results if e['is_relevant'] and e['confidence'] == 0.0)

        logger.info("News enrichment complete:")
        logger.info(f"  - Total processed: {len(enrichment_results)}")
        logger.info(f"  - Relevant articles: {relevant_articles}")
        logger.info(f"  - Successful enrichments: {successful_enrichments}")
        logger.info(f"  - Invalid/low-confidence responses: {invalid_responses}")
        logger.info(f"  - Skipped non-financial: {skipped_non_financial}")
        logger.info(f"  - Failed: {failed_enrichments}")
        
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
        
        # Optionally reprocess the entire database
        reprocessed_count = 0
        if reprocess_existing:
            logger.info("\n[6/6] Reprocessing entire news database...")
            existing_news = db.get_all_news()
            reprocess_texts = [f"{row.get('title', '')} {row.get('content', '')}".strip() for row in existing_news]
            reprocess_results = enrich_batch(reprocess_texts)
            updated_rows = []
            for row, enrichment in zip(existing_news, reprocess_results):
                updated_rows.append({
                    "url": row.get("url", ""),
                    "is_relevant": enrichment["is_relevant"],
                    "sentiment": enrichment["sentiment"],
                    "confidence": enrichment["confidence"],
                    "segments": enrichment["segments"],
                    "tickers": enrichment["tickers"]
                })
            reprocessed_count = db.update_news_batch(updated_rows)
            logger.info(f"Reprocessed database count: {reprocessed_count}")
        
        # Get final count
        final_count = db.get_news_count()
        
        logger.info(f"Database stats: {initial_count} -> {final_count} records")
        logger.info(f"Inserted in this run: {inserted_count}")
        if reprocess_existing:
            logger.info(f"Reprocessed existing records: {reprocessed_count}")
        
        # Pipeline summary
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline execution summary:")
        logger.info(f"  - Total fetched: {len(raw_news)}")
        logger.info(f"  - After deduplication: {len(deduplicated_news)}")
        logger.info(f"  - Valid entries: {len(valid_news)}")
        logger.info(f"  - Sentiment enrichments: {successful_enrichments} successful, {failed_enrichments} failed")
        logger.info(f"  - Successfully inserted: {inserted_count}")
        logger.info(f"  - Database total records: {final_count}")
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the financial news ingestion pipeline.")
    parser.add_argument(
        '--reprocess-existing',
        action='store_true',
        help='Reprocess the entire news database after inserting new articles.'
    )
    args = parser.parse_args()

    run_pipeline(reprocess_existing=args.reprocess_existing)
