from __future__ import annotations

"""Batch: embed existing lessons without embeddings and upsert vector.

Usage:
  python -m pipeline.ops.embed_lessons --scope global --limit 200
"""

import argparse
import json
from typing import Any, Dict

from ..agents.embeddings import embed_text
from ..infra.db import get_conn, upsert_agent_lesson_embedding_by_hash


def _fetch_without_embeddings(scope: str, limit: int) -> list[Dict[str, Any]]:
    sql = (
        "SELECT created_at, scope, lesson_text, meta FROM agent_lessons "
        "WHERE scope=%s AND (lesson_embedding IS NULL) ORDER BY created_at DESC LIMIT %s"
    )
    out: list[Dict[str, Any]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (scope, int(limit)))
            rows = cur.fetchall() or []
            for created_at, scope_v, text, meta in rows:
                out.append({
                    "created_at": created_at,
                    "scope": scope_v,
                    "text": text,
                    "meta": meta or {},
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="global")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    rows = _fetch_without_embeddings(args.scope, args.limit)
    if not rows:
        print(json.dumps({"status": "ok", "embedded": 0}))
        return
    cnt = 0
    for r in rows:
        text = str(r.get("text") or "")
        meta = r.get("meta") or {}
        h = meta.get("hash") or None
        if not h:
            # derive from text
            h = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
        vec = embed_text(text)
        upsert_agent_lesson_embedding_by_hash(args.scope, str(h), vec)
        cnt += 1
    print(json.dumps({"status": "ok", "embedded": cnt}))


if __name__ == "__main__":
    main()

