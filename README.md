# B3 Market Feeling Detector - Financial News Ingestion Pipeline with Sentiment Analysis

A robust, production-ready Python pipeline for ingesting financial news from Brazilian RSS feeds with AI-powered sentiment analysis. The pipeline collects, processes, stores news articles, and enriches them with sentiment classification using OpenAI's GPT models.

## Overview

This project implements a complete data engineering solution for:
- **Real-time ingestion** of financial news from 3 major Brazilian sources
- **Data cleaning and normalization** with HTML tag stripping and text processing
- **AI-powered sentiment analysis** using OpenAI GPT models for market sentiment detection
- **Dual storage** with raw JSON files and SQLite database for different use cases
- **Deduplication** to prevent storing identical articles
- **Comprehensive logging** for monitoring and debugging
- **Error handling** with retry logic for network resilience

## Project Structure

```
b3-market-feeling-detector/
├── src/                     # Source code directory
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── fetch_news.py    # RSS feed fetching with retry logic
│   │   └── sources.py       # Configured RSS feed sources
│   ├── processing/
│   │   ├── __init__.py
│   │   └── clean_news.py    # Data cleaning and normalization
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── sentiment.py     # AI-powered sentiment analysis
│   └── storage/
│       ├── __init__.py
│       ├── save_raw.py      # Raw JSON file storage
│       └── database.py      # SQLite database management
├── tests/                   # Test suite
│   ├── conftest.py         # Pytest configuration and fixtures
│   ├── test_ingestion.py   # Tests for ingestion module
│   ├── test_processing.py  # Tests for processing module
│   ├── test_storage.py     # Tests for storage module
│   ├── test_database.py    # Tests for database module
│   └── test_nlp.py         # Tests for sentiment analysis module
├── data/
│   ├── raw/                # Raw JSON files (daily)
│   └── processed/          # Processed data (optional)
├── main.py                 # Pipeline orchestration
├── requirements.txt        # Python dependencies
├── pytest.ini             # Pytest configuration
├── .gitignore            # Git ignore file
├── README.md             # This file
└── pipeline.log          # Execution logs (created on first run)
```

## Features

### Ingestion Module
- **Feedparser-based** RSS feed parsing with error tolerance
- **Retry logic** with exponential backoff for network resilience
- **Multiple source support** defined in `sources.py`
- **Robust error handling** for parsing failures

### Processing Module
- **HTML tag removal** from article summaries
- **Text normalization** (whitespace, special characters)
- **Date standardization** to ISO 8601 format
- **Entry validation** to ensure data quality

### NLP Module (Sentiment Analysis)
- **OpenAI GPT integration** for financial sentiment classification
- **Portuguese language support** for Brazilian financial news
- **Confidence scoring** (0-1) for sentiment reliability
- **Cost optimization** with text truncation and caching
- **Batch processing** for efficient API usage
- **Error handling** with graceful fallback to neutral sentiment

### Storage Module

**Raw Storage (JSON):**
- Daily JSON files with format: `news_YYYY-MM-DD.json`
- Automatic append mode (doesn't overwrite existing data)
- Text deduplication by URL

**Database (SQLite):**
- Table: `news` with auto-increment primary key
- Unique constraint on URL to prevent duplicates
- Indexed columns: `source`, `published_at`, `collected_at`
- Sentiment and confidence scores for each article
- Supports quick lookups by source, date range, or sentiment

### Database Schema

```sql
CREATE TABLE news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT,
    source TEXT NOT NULL,
    published_at TEXT,
    url TEXT NOT NULL UNIQUE,
    collected_at TEXT NOT NULL,
    sentiment TEXT,           -- 'positivo', 'negativo', or 'neutro'
    confidence REAL           -- Float between 0 and 1
)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenAI API key (for sentiment analysis)

### Step 1: Clone or Create Project

```bash
cd b3-market-feeling-detector
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure OpenAI API Key

Get your API key from [OpenAI Platform](https://platform.openai.com/account/api-keys).

Set the environment variable:

```bash
# On Windows (PowerShell)
$env:OPENAI_API_KEY = "your-api-key-here"

# On Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# On macOS/Linux
export OPENAI_API_KEY="your-api-key-here"
```

For permanent configuration, add to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Fetch news from all configured RSS sources
2. Deduplicate entries
3. Clean and validate the data
4. **Classify sentiment** using OpenAI GPT models
5. Save raw data to JSON file in `data/raw/`
6. Insert cleaned data with sentiment scores into SQLite database
7. Log results to `pipeline.log`

### Example Output

```
============================================================
Starting news ingestion pipeline
============================================================

[1/6] Fetching news from RSS sources...
Configured sources: 3
INFO:ingestion.fetch_news:Fetching feed from InfoMoney: https://www.infomoney.com.br/feed/
INFO:ingestion.fetch_news:Successfully fetched 50 entries from InfoMoney
INFO:ingestion.fetch_news:Total news collected: 150

[2/6] Deduplicating news articles...
Deduplicated 5 duplicate entries

[3/6] Cleaning and processing news data...
Cleaned 150 out of 150 news entries
Valid news entries: 150

[4/6] Classifying sentiment for news articles...
Sentiment classification complete:
  - Total processed: 150
  - Successful: 148
  - Failed: 2

[5/6] Saving raw data to JSON files...
Saved 147 new news entries to data/raw/news_2024-03-26.json (total: 157)

[6/6] Inserting cleaned data into SQLite database...
Database insert complete: 145 inserted, 5 skipped (duplicates)

============================================================
Pipeline execution summary:
  - Total fetched: 150
  - After deduplication: 150
  - Valid entries: 150
  - Sentiment classifications: 148 successful, 2 failed
  - Successfully inserted: 145
  - Database total records: 457
============================================================
```

## Data Flow

```
RSS Feeds (3 sources) 
    ↓
fetch_news.py (Fetch & Parse)
    ↓
deduplicate_news (URL-based dedup)
    ↓
clean_news.py (Clean & Validate)
    ↓
sentiment.py (AI Sentiment Analysis)
    ↓
┌─────────────────────────────┐
│                             │
save_raw.py (JSON)     database.py (SQLite)
│                             │
data/raw/news_YYYY-MM-DD.json  data/news.db
└─────────────────────────────┘
```

## Configuration

### Adding New RSS Sources

Edit `ingestion/sources.py`:

```python
FINANCIAL_SOURCES: List[RSSSource] = [
    RSSSource(
        name="Your Source",
        url="https://your-feed-url/rss/"
    ),
    # ... existing sources
]
```

### Database Location

Default: `data/news.db`

To change, modify in `main.py`:
```python
db = NewsDatabase(db_path="path/to/your/database.db")
```

### Sentiment Analysis Configuration

The sentiment analysis uses the following defaults:
- **Model**: `gpt-4o-mini` (cost-effective)
- **Max text length**: 1000 characters
- **Max retries**: 2 attempts
- **Cache**: In-memory for duplicate text avoidance

To modify settings, edit `nlp/sentiment.py`:

```python
# Change model
MODEL_NAME = "gpt-4"  # More accurate but expensive

# Change text limit
MAX_TEXT_LENGTH = 2000  # Longer context

# Change retry count
MAX_RETRIES = 3  # More resilient
```

### Retry Configuration

Edit retry settings in `ingestion/fetch_news.py`:

```python
def setup_session(retries: int = 3, timeout: int = 10):
    # Customize retry count and timeout
```

## Logs and Monitoring

### Log Files
- **Console**: Real-time pipeline execution status
- **pipeline.log**: Complete execution history with timestamps

### Sample Log Entries
```
2024-03-26 14:23:45,123 - ingestion.fetch_news - INFO - Fetching feed from InfoMoney
2024-03-26 14:23:47,456 - ingestion.fetch_news - INFO - Successfully fetched 50 entries
2024-03-26 14:23:48,789 - storage.save_raw - INFO - Saved 47 new news entries to data/raw/news_2024-03-26.json
```

## Running Pipeline Automatically

### Option 1: Windows Task Scheduler

Create a batch file `run_pipeline.bat`:
```batch
@echo off
cd C:\path\to\b3-market-feeling-detector
venv\Scripts\activate
python main.py
```

Schedule with Task Scheduler to run daily.

### Option 2: Linux/macOS Cron

Add to crontab (`crontab -e`):
```bash
# Run daily at 9 AM
0 9 * * * cd /path/to/b3-market-feeling-detector && /path/to/venv/bin/python main.py
```

### Option 3: Windows Scheduled Task (Direct)

```powershell
# PowerShell as Administrator
$trigger = New-ScheduledTaskTrigger -Daily -At 9am
$action = New-ScheduledTaskAction -Execute "C:\path\to\venv\Scripts\python.exe" -Argument "C:\path\to\main.py"
Register-ScheduledTask -TaskName "Financial News Pipeline" -Trigger $trigger -Action $action
```

## Database Queries

### Connect to Database

```bash
sqlite3 data/news.db
```

### Useful Queries

```sql
-- Total news count
SELECT COUNT(*) FROM news;

-- News by source
SELECT source, COUNT(*) as count FROM news GROUP BY source;

-- Latest 10 articles
SELECT title, source, published_at FROM news ORDER BY published_at DESC LIMIT 10;

-- News from last 24 hours
SELECT * FROM news WHERE collected_at > datetime('now', '-1 day');

-- Sentiment distribution
SELECT sentiment, COUNT(*) as count FROM news GROUP BY sentiment;

-- High confidence positive sentiment
SELECT title, source, confidence FROM news 
WHERE sentiment = 'positivo' AND confidence > 0.8 
ORDER BY confidence DESC LIMIT 10;

-- Sentiment by source
SELECT source, sentiment, COUNT(*) as count 
FROM news GROUP BY source, sentiment 
ORDER BY source, sentiment;

-- Average confidence by sentiment
SELECT sentiment, AVG(confidence) as avg_confidence, COUNT(*) as count
FROM news GROUP BY sentiment;

-- Recent negative sentiment news
SELECT title, source, published_at, confidence FROM news 
WHERE sentiment = 'negativo' AND published_at > datetime('now', '-7 days')
ORDER BY published_at DESC;

-- Duplicate detection (should be none due to unique constraint)
SELECT url, COUNT(*) FROM news GROUP BY url HAVING COUNT(*) > 1;
```

## API Costs and Optimization

### OpenAI API Usage

The sentiment analysis uses OpenAI's `gpt-4o-mini` model, which is cost-effective for this use case:

- **Cost per 1K tokens**: ~$0.00015 (input) + ~$0.0006 (output)
- **Typical usage**: ~200-400 tokens per news article
- **Estimated cost**: ~$0.0001-0.0002 per article

For 150 articles/day: **~$0.02/day** or **~$7.50/month**

### Cost Optimization Features

- **Text truncation**: Limited to 1000 characters to reduce token usage
- **In-memory caching**: Avoids re-processing identical text
- **Batch processing**: Efficient API usage patterns
- **Error handling**: Graceful fallback prevents wasted API calls

### Monitoring API Usage

Check your OpenAI usage dashboard at [platform.openai.com](https://platform.openai.com/usage) to monitor costs.

### Reducing Costs

```python
# Skip sentiment analysis for existing articles
# (Already implemented - only processes new articles)

# Use cheaper model if accuracy allows
MODEL_NAME = "gpt-3.5-turbo"  # ~80% cheaper than gpt-4o-mini

# Reduce text length further
MAX_TEXT_LENGTH = 500  # Shorter context
```

## Troubleshooting

### Issue: `No module named 'feedparser'`
**Solution**: Ensure virtual environment is activated and dependencies installed
```bash
pip install -r requirements.txt
```

### Issue: Database locked error
**Solution**: Close any other connections to `data/news.db` and try again

### Issue: No entries from a particular source
**Solution**: Check `pipeline.log` for parsing errors, the feed might have changed format

### Issue: Slow pipeline execution
**Solution**: 
- Check network connectivity
- Verify RSS feed availability
- Check disk space for JSON files

### Issue: OpenAI API key not configured
**Solution**: Set the `OPENAI_API_KEY` environment variable
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Issue: Sentiment analysis failing with API errors
**Solution**: 
- Verify API key is valid and has credits
- Check OpenAI service status
- Review `pipeline.log` for specific error messages
- The pipeline will continue with neutral sentiment as fallback

### Issue: High API costs
**Solution**:
- Monitor usage at [platform.openai.com](https://platform.openai.com/usage)
- Consider reducing `MAX_TEXT_LENGTH` in `sentiment.py`
- Use caching is already enabled to avoid duplicate processing

### Issue: All sentiment results are "neutro"
**Solution**:
- Check if API calls are succeeding in logs
- Verify the prompt format in `sentiment.py`
- Test manually: `python -c "from src.nlp.sentiment import classify_sentiment; print(classify_sentiment('test text'))"`

## Architecture Decisions

### Why Two Storage Methods?

1. **Raw JSON** (`data/raw/`):
   - Maintains audit trail of all fetched data
   - Supports time-travel queries
   - Easy to share or archive
   - Daily files for partitioning

2. **SQLite** (`data/news.db`):
   - Fast indexed queries
   - ACID compliance
   - Duplicate prevention
   - Structured data for analytics

### Why Deduplicate at Multiple Levels?

1. **Fetch level**: Removes duplicates from single RSS fetch
2. **Raw save level**: Prevents duplicate URLs in JSON
3. **Database level**: Unique constraint on URL

This ensures data quality despite source redundancy.

### Why OpenAI for Sentiment Analysis?

1. **Portuguese Language Support**: GPT models handle Brazilian Portuguese financial terminology well
2. **Context Understanding**: Can interpret financial impact and market sentiment nuances
3. **Confidence Scoring**: Provides reliability metrics for sentiment classifications
4. **Cost-Effective**: `gpt-4o-mini` balances accuracy and cost
5. **Production Ready**: Reliable API with good error handling

### Why In-Memory Caching?

1. **Cost Reduction**: Avoids re-processing identical news text
2. **Performance**: Faster pipeline execution for duplicate content
3. **API Efficiency**: Reduces unnecessary API calls
4. **Simple Implementation**: Dict-based cache works well for batch processing

### Why Strict JSON Response Format?

1. **Reliability**: Structured output ensures consistent parsing
2. **Validation**: Easy to validate sentiment values and confidence ranges
3. **Error Handling**: Clear fallback for malformed responses
4. **Consistency**: Same format across all API calls

## Code Quality

- **Type hints** throughout for better IDE support
- **Comprehensive docstrings** for maintainability
- **Modular design** for easy testing and extension
- **Exception handling** at critical points
- **Logging** for observability and debugging
- **API error handling** with retry logic and fallbacks
- **Cost optimization** with caching and text limits

## Testing

### Pytest Suite (57 Unit Tests)

The project includes comprehensive pytest coverage for all modules.

#### Run All Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run tests with shorter output
python -m pytest tests/
```

#### Run Specific Test Files

```bash
# Test ingestion module
python -m pytest tests/test_ingestion.py -v

# Test processing module
python -m pytest tests/test_processing.py -v

# Test storage module
python -m pytest tests/test_storage.py -v

# Test database module
python -m pytest tests/test_database.py -v

# Test sentiment analysis module
python -m pytest tests/test_nlp.py -v
```

#### Test Coverage

The test suite covers:
- **Ingestion**: RSS source configuration, feed deduplication
- **Processing**: HTML tag stripping, text normalization, date standardization, validation
- **NLP**: Sentiment classification, API error handling, caching, batch processing
- **Storage**: JSON file operations, deduplication
- **Database**: CRUD operations, duplicate prevention, querying

Current coverage: **57 tests, all passing**

```
tests/test_database.py ............ 8 tests passed
tests/test_ingestion.py ........... 13 tests passed
tests/test_nlp.py ................ 13 tests passed
tests/test_processing.py .......... 17 tests passed
tests/test_storage.py ............ 6 tests passed

Total: 57 passed
```

#### Test Fixtures

The project uses pytest fixtures for:
- Temporary data directories
- Sample news data
- Database instances
- Raw RSS data

See `tests/conftest.py` for all available fixtures.

### Manual Testing

```bash
# Test fetching from updated import path
python -c "from src.ingestion.fetch_news import fetch_all_news; from src.ingestion.sources import get_sources; print(fetch_all_news(get_sources()))"

# Test database
python -c "from src.storage.database import NewsDatabase; db = NewsDatabase(); print(f'Total records: {db.get_news_count()}')"

# Test cleaning
python -c "from src.processing.clean_news import clean_news_batch; news = [{'title': '<h1>Test</h1>', 'summary': 'Test', 'link': 'http://test.com', 'source': 'test', 'published_at': None, 'collected_at': '2024-03-26T00:00:00'}]; print(clean_news_batch(news))"

# Test sentiment analysis (requires OPENAI_API_KEY)
python -c "from src.nlp.sentiment import classify_sentiment; result = classify_sentiment('Petrobras announces record profits'); print(result)"

# Test batch sentiment analysis
python -c "from src.nlp.sentiment import classify_batch; texts = ['Good news', 'Bad news', 'Neutral update']; results = classify_batch(texts); print(results)"
```

## Future Enhancements

- [x] Sentiment analysis of article titles/content
- [ ] Topic modeling with clustering
- [ ] Time-series trend analysis
- [ ] Docker containerization
- [ ] REST API for query interface
- [ ] Data export to CSV/Parquet
- [ ] Automated unit tests
- [ ] Airflow/Prefect orchestration
- [ ] Real-time sentiment dashboard
- [ ] Multi-language support for international markets

## License

See LICENSE file in project root.

## Support

For issues or questions:
1. Check `pipeline.log` for error details
2. Review database schema with `sqlite3 data/news.db ".schema"`
3. Test individual modules in isolation
4. Verify RSS feed URLs are still active

---

**Last Updated**: March 2024
**Python Version**: 3.8+
**Maintained**: Yes
