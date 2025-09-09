#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import requests


def main():
    base = os.getenv("FLOWISE_BASE_URL", "http://flowise:3000")
    url = base.rstrip("/") + "/api/v1/ping"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    print("flowise_ok")


if __name__ == "__main__":
    main()

