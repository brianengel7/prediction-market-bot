#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

MINUTES="${1:-10}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR=".collector_runs/$RUN_ID"

mkdir -p "$LOG_DIR"

# Keep runtime logs out of Git without changing
# the repository's tracked .gitignore.
if ! grep -qxF '.collector_runs/' .git/info/exclude; then
    echo '.collector_runs/' >> .git/info/exclude
fi

BOOK_PID=""
WEATHER_PID=""

cleanup() {
    trap - EXIT INT TERM
    if [[ -n "$BOOK_PID" ]]; then
        kill "$BOOK_PID" 2>/dev/null || true
    fi
    if [[ -n "$WEATHER_PID" ]]; then
        kill "$WEATHER_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "========================================="
echo "KALSHI WEATHER LATENCY RESEARCH"
echo "========================================="
echo "Start UTC: $(date -u --iso-8601=seconds)"
echo "Duration: $MINUTES minutes"
echo "City: Miami"
echo "Log directory: $LOG_DIR"
echo "Trading: DISABLED"
echo

# Collector 1: Live Kalshi order books
python -u -m src.kalshi.latency_scanner_v1 \
    --series KXTEMPMIAH \
    --max-markets 20 \
    --sample-seconds 5 \
    --minutes "$MINUTES" \
    > "$LOG_DIR/orderbook.log" 2>&1 &

BOOK_PID=$!

# Collector 2: Official weather index
python -u -m src.kalshi.weather_index_collector \
    --city miami \
    --interval 20 \
    --last-sec 600 \
    --minutes "$MINUTES" \
    > "$LOG_DIR/weather.log" 2>&1 &

WEATHER_PID=$!

echo "Order-book PID: $BOOK_PID"
echo "Weather PID: $WEATHER_PID"
echo
echo "Both collectors started."
echo "Monitoring their output every 30 seconds."
echo

# Periodically show recent activity from each feed.
while kill -0 "$BOOK_PID" 2>/dev/null &&
      kill -0 "$WEATHER_PID" 2>/dev/null; do

    sleep 30

    echo "[$(date -u +%H:%M:%S) UTC]"

    echo "ORDER BOOK:"
    tail -n 2 "$LOG_DIR/orderbook.log" || true

    echo "WEATHER:"
    tail -n 2 "$LOG_DIR/weather.log" || true

    echo "-----------------------------------------"
done

BOOK_RC=0
WEATHER_RC=0

wait "$BOOK_PID" || BOOK_RC=$?
wait "$WEATHER_PID" || WEATHER_RC=$?

echo
echo "========================================="
echo "COLLECTION SUMMARY"
echo "========================================="
echo "End UTC: $(date -u --iso-8601=seconds)"
echo "Order-book exit code: $BOOK_RC"
echo "Weather exit code: $WEATHER_RC"

echo
echo "ORDER-BOOK LOG:"
tail -n 8 "$LOG_DIR/orderbook.log"

echo
echo "WEATHER LOG:"
tail -n 8 "$LOG_DIR/weather.log"

echo
echo "Potential arbitrage alerts:"
grep -c 'POTENTIAL ARB' "$LOG_DIR/orderbook.log" || true

echo
echo "Logs saved to: $LOG_DIR"

if [[ "$BOOK_RC" -ne 0 || "$WEATHER_RC" -ne 0 ]]; then
    echo "WARNING: At least one collector exited with an error."
    exit 1
fi

echo "Both collectors completed successfully."
