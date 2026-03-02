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
if [ -f .env ]; then
    echo "[Info] Loading environment from .env"
    export $(grep -v '^#' .env | xargs)
fi

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

$PYTHON_CMD scripts/train_online.py "$SYMBOL" \
    --interval "$INTERVAL" \
    --bars "$BARS" \
    --warmup-bars "$WARMUP_BARS" \
    --sequence-length "$SEQ_LENGTH" \
    --output "signals_${SYMBOL,,}.csv" \
    --model-output "model_${SYMBOL,,}.joblib"

echo ""
echo "[Success] Training pipeline completed."
echo "[Output] Signals: signals_${SYMBOL,,}.csv"
echo "[Output] Model:   model_${SYMBOL,,}.joblib"
echo "============================================================"

