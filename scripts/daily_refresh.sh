#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# NHL Pipeline — Daily Refresh
#
# Runs automatically via cron. Fetches today's schedule, updates rosters,
# pulls all player game logs, fetches team/goalie stats, and retrains
# the scoring model.
#
# Logs are written to PROJECT_DIR/logs/daily_YYYY-MM-DD.log
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths (edit these if your setup differs) ──────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${PROJECT_DIR}/logs"
MODEL_PATH="${PROJECT_DIR}/data/model.pkl"

mkdir -p "$LOG_DIR"

DATE=$(date +%F)
LOGFILE="${LOG_DIR}/daily_${DATE}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

# ── Main ──────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

log "═══ NHL daily refresh starting ═══"

# 1. Fetch today's schedule + refresh all 32 team rosters
log "Step 1/4: Fetching schedule and rosters..."
if $PYTHON run_pipeline.py daily >> "$LOGFILE" 2>&1; then
    log "  ✓ Schedule and rosters updated"
else
    log "  ✗ Schedule/roster fetch failed (exit $?) — continuing anyway"
fi

# 2. Pull game logs for all teams (full season, upserts only new data)
log "Step 2/4: Fetching player game logs (all teams)..."
if $PYTHON run_pipeline.py gamelogs >> "$LOGFILE" 2>&1; then
    log "  ✓ Game logs updated"
else
    log "  ✗ Game log fetch failed (exit $?) — continuing anyway"
fi

# 3. Fetch team stats + goalie logs for all completed games this season
log "Step 3/4: Fetching team and goalie stats..."
if $PYTHON run_pipeline.py gamestats --season >> "$LOGFILE" 2>&1; then
    log "  ✓ Team/goalie stats updated"
else
    log "  ✗ Team/goalie stats fetch failed (exit $?) — continuing anyway"
fi

# 4. Retrain the model on the updated data
log "Step 4/4: Retraining model..."
if $PYTHON run_pipeline.py train --model "$MODEL_PATH" >> "$LOGFILE" 2>&1; then
    log "  ✓ Model retrained → ${MODEL_PATH}"
else
    log "  ✗ Model training failed (exit $?)"
fi

log "═══ NHL daily refresh complete ═══"

# ── Cleanup old logs (keep last 30 days) ──────────────────────────────
find "$LOG_DIR" -name "daily_*.log" -mtime +30 -delete 2>/dev/null || true
