#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from loguru import logger


def main(slot: str | None = None):
    slot = slot or os.getenv("SLOT", "manual")
    try:
        # Try importing pipeline directly (if repo mounted into Windmill)
        if "/app/src" not in sys.path:
            sys.path.append("/app/src")
        from pipeline.orchestration.predict_release import predict_release  # type: ignore

        predict_release(slot)
        return
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Direct import failed: {e}")

    # Fallback: run via docker compose (requires docker available to Windmill)
    try:
        import subprocess

        cmd = [
            "docker",
            "compose",
            "run",
            "--rm",
            "-e",
            f"SLOT={slot}",
            "pipeline",
            "-m",
            "pipeline.orchestration.predict_release",
        ]
        subprocess.check_call(cmd)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Could not execute predict_release. Ensure either the repo is mounted into Windmill at /app/src "
            "or docker is available inside Windmill to run 'docker compose run pipeline'."
        )
        raise


if __name__ == "__main__":
    main()
