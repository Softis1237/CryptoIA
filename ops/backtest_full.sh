#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$(pwd)/services/pipeline/src"
fi

PRICE_FILE="${1:-data/btc_2020_2024.parquet}"
OUT_DIR="${2:-artifacts/backtest}";
mkdir -p "$OUT_DIR"

python services/pipeline/src/pipeline/cli/backtest.py \
  "$PRICE_FILE" \
  --start 2022-01-01 --end 2023-12-31 \
  --strategy pipeline.trading.backtest.strategies:MovingAverageCrossStrategy \
  --strategy-param fast=12 \
  --strategy-param slow=26 \
  --report "$OUT_DIR/summary.json"

echo "Base MA strategy completed. Replace --strategy with кастомной реализацией, которая вызывает MasterAgent, чтобы оценить полный конвейер."

echo "Backtest finished. Report saved to $OUT_DIR/summary.json"
