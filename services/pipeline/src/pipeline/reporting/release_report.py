from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..infra.s3 import upload_bytes


def save_release_report(
    *,
    run_id: str,
    slot: str,
    regime: Dict[str, Any],
    neighbors: List[Dict[str, Any]],
    ensemble_4h: Dict[str, Any],
    ensemble_12h: Dict[str, Any],
    per_model_4h: Dict[str, Any],
    per_model_12h: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    trade_card: Dict[str, Any],
    news_top: List[Dict[str, Any]],
    artifacts: Dict[str, str],
) -> str:
    """Compose and upload a single JSON report with all release artifacts.

    Returns S3 URI of the saved report.
    """
    now = datetime.now(timezone.utc)
    report = {
        "run_id": run_id,
        "slot": slot,
        "generated_at": now.isoformat(timespec="seconds"),
        "regime": regime,
        "neighbors": neighbors,
        "predictions": {
            "4h": {"ensemble": ensemble_4h, "per_model": per_model_4h},
            "12h": {"ensemble": ensemble_12h, "per_model": per_model_12h},
        },
        "scenarios": scenarios,
        "trade_card": trade_card,
        "news_top": news_top,
        "artifacts": artifacts,
    }

    body = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    date_key = now.strftime("%Y-%m-%d")
    path = f"runs/{date_key}/{slot}/release.json"
    return upload_bytes(path, body, content_type="application/json")
