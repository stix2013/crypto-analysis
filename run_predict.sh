#!/bin/bash

# run_predict.sh - Helper script for model inference
#
# Usage: ./run_predict.sh [SYMBOL] [INTERVAL] [BARS]
# Example: ./run_predict.sh BTCUSDT 1h 200

set -e

# --- Default Configuration ---
SYMBOL=${1:-"ETHUSDT"}
INTERVAL=${2:-"15m"}
BARS=${3:-200}

# --- Environment Setup ---
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -f "$BASE_DIR/.env" ]; then
    echo "[Info] Loading environment from .env"
    export $(grep -v '^#' "$BASE_DIR/.env" | xargs)
fi

# Ensure output directory exists
mkdir -p "$BASE_DIR/signals"

# Determine Python command
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

# Ensure src is in PYTHONPATH
export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"

# --- Execution ---
echo "============================================================"
echo " Starting Crypto Analysis Prediction"
echo " Symbol:   $SYMBOL"
echo " Interval: $INTERVAL"
echo " Bars:     $BARS"
echo "============================================================"

# --- Output ---
OUTPUT_SIGNALS="predict_${SYMBOL,,}_${INTERVAL}.csv"

$PYTHON_CMD "$BASE_DIR/scripts/predict.py" "$SYMBOL" \
    --interval "$INTERVAL" \
    --bars "$BARS" \
    --output "$BASE_DIR/signals/${OUTPUT_SIGNALS}"

echo ""
echo "[Success] Prediction completed."
echo "[Output] Signals: ./signals/${OUTPUT_SIGNALS}"
echo "============================================================"
