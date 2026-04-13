#!/bin/sh
set -e

# Export current environment variables (passed by Docker / docker-compose) so
# that the cron daemon can make them available to the scheduled job.
# Debian cron reads /etc/environment before executing each job.
printenv | grep -v '^_' > /etc/environment
chmod 600 /etc/environment

exec cron -f
