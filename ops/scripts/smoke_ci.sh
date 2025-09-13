#!/usr/bin/env bash
set -euo pipefail

echo "Validating docker compose config..."
docker compose -f docker-compose.yml config >/dev/null

echo "Python import smoke..."
python - << 'PY'
import sys
sys.path.insert(0, 'services/pipeline/src')
import pipeline
import pipeline.orchestration.agent_flow as af
print('OK:', pipeline.__doc__)
print('OK:', af.__name__)
PY

echo "Done."

