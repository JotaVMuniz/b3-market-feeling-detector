FROM python:3.11-slim

# Install cron daemon (used by the pipeline service)
RUN apt-get update && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Register the daily pipeline cron job
RUN cp crontab /etc/cron.d/pipeline \
    && chmod 0644 /etc/cron.d/pipeline

# Make the cron entrypoint executable
RUN chmod +x entrypoint.sh

# Streamlit dashboard port
EXPOSE 8501
ENV PYTHONUNBUFFERED=1

# Default: run the Streamlit dashboard.
# Override with ENTRYPOINT/CMD in docker-compose to run the cron pipeline.
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
