#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHONPATH=services/pipeline/src \
  python services/pipeline/ops/forecast_quality_metrics.py --hours 168 --horizon 4h
