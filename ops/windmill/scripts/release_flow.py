from __future__ import annotations

"""Windmill script entrypoint to run the release flow.

Upload this script into Windmill as a Python script and schedule it.
Requires the workspace to have network access to Postgres/MinIO and
environment variables matching docker-compose settings.
"""

import os
import sys


def main(slot: str | None = None):
    if '/workspace/services/pipeline/src' not in sys.path:
        # Adjust this path depending on how you mount the repo into Windmill
        sys.path.append('/workspace/services/pipeline/src')
    from pipeline.orchestration.agent_flow import run_release_flow
    run_release_flow(slot=slot or os.environ.get('SLOT', 'windmill'))


if __name__ == '__main__':
    main()

