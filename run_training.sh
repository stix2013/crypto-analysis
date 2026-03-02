#!/bin/bash

# run_training.sh - Helper script for online learning training
#
# Usage: ./run_training.sh [SYMBOL] [INTERVAL] [BARS]
# Example: ./run_training.sh BTCUSDT 1h 2000

set -e

# --- Default Configuration ---
SYMBOL=${1:-"ETHUSDT"}
INTERVAL=${2:-"15m"}
BARS=${3:-5000}
WARMUP_BARS=1000
SEQ_LENGTH=60

# --- Environment Setup ---
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -f "$BASE_DIR/.env" ]; then
    echo "[Info] Loading environment from .env"
    export $(grep -v '^#' "$BASE_DIR/.env" | xargs)
fi

# Ensure output directories exist
mkdir -p "$BASE_DIR/signals" "$BASE_DIR/models"

# Determine Python command
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

# --- Execution ---
echo "============================================================"
echo " Starting Crypto Analysis Training Pipeline"
echo " Symbol:   $SYMBOL"
echo " Interval: $INTERVAL"
echo " Bars:     $BARS"
echo "============================================================"

# --- Output ---
OUTPUT_SIGNALS="signals_${INTERVAL}_${SYMBOL,,}.csv"
OUTPUT_MODEL="model_${INTERVAL}_${SYMBOL,,}.joblib"

$PYTHON_CMD "$BASE_DIR/scripts/train_online.py" "$SYMBOL" \
    --interval "$INTERVAL" \
    --bars "$BARS" \
    --warmup-bars "$WARMUP_BARS" \
    --sequence-length "$SEQ_LENGTH" \
    --output "$BASE_DIR/signals/${OUTPUT_SIGNALS}" \
    --model-output "$BASE_DIR/models/${OUTPUT_MODEL}"

echo ""
echo "[Success] Training pipeline completed."
echo "[Output] Signals: ./signals/${OUTPUT_SIGNALS}"
echo "[Output] Model:   ./models/${OUTPUT_MODEL}"
echo "============================================================"
