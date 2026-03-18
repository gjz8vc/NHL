#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Install the NHL daily refresh cron job.
#
# Run this ONCE:   bash scripts/install_cron.sh
#
# By default it schedules the refresh at 8:00 AM local time daily.
# Change CRON_SCHEDULE below to adjust.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${PROJECT_DIR}/scripts/daily_refresh.sh"
CRON_SCHEDULE="0 8 * * *"   # 8:00 AM every day

# Ensure the script is executable
chmod +x "$SCRIPT"

# Add to crontab (avoids duplicates)
CRON_LINE="${CRON_SCHEDULE} ${SCRIPT}"
( crontab -l 2>/dev/null | grep -v "daily_refresh.sh" ; echo "$CRON_LINE" ) | crontab -

echo "Cron job installed:"
echo "  Schedule : ${CRON_SCHEDULE}  (8:00 AM daily)"
echo "  Script   : ${SCRIPT}"
echo ""
echo "Verify with:  crontab -l"
echo "Remove with:  crontab -l | grep -v daily_refresh.sh | crontab -"
