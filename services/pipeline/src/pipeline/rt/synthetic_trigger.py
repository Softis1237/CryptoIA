from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from .queue import publish_trigger


def main() -> None:
    p = argparse.ArgumentParser(description="Publish a synthetic RT trigger")
    p.add_argument("type", help="Trigger type, e.g. VOL_SPIKE, DELTA_SPIKE, NEWS, L2_WALL, L2_IMBALANCE")
    p.add_argument("--meta", help="JSON meta payload", default="{}")
    p.add_argument("--ts", help="Epoch seconds (optional)", default=None)
    args = p.parse_args()
    meta: Dict[str, Any] = {}
    try:
        meta = json.loads(args.meta)
    except Exception:
        meta = {}
    ev = {"type": args.type.upper(), "ts": None if args.ts is None else int(args.ts), "symbol": meta.get("symbol") or "BTC/USDT", "meta": meta}
    ok = publish_trigger(ev)
    print(json.dumps({"ok": ok, "event": ev}, ensure_ascii=False))


if __name__ == "__main__":
    main()

