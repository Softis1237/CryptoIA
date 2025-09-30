#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHONPATH=services/pipeline/src \
  python services/pipeline/ops/recompute_agent_performance.py --days 180
