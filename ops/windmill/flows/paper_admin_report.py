#!/usr/bin/env python3
from __future__ import annotations

import sys


def main():
    if "/app/src" not in sys.path:
        sys.path.append("/app/src")
    from pipeline.trading.paper_trading import admin_report, admin_weekly_report  # type: ignore

    admin_report()
    admin_weekly_report()


if __name__ == "__main__":
    main()
