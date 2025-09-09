#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from loguru import logger


def _ensure_path():
    if "/app/src" not in sys.path:
        sys.path.append("/app/src")


def main():
    _ensure_path()
    from pipeline.ops.feedback_metrics import run as run_feedback

    window_days = int(os.getenv("FEEDBACK_WINDOW_DAYS", "7"))
    res = run_feedback(window_days=window_days)
    print(json.dumps(res))


if __name__ == "__main__":
    main()

