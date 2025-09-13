from __future__ import annotations

"""Adapt model trust weights from paper PnL (lightweight, best-effort).

For each closed paper trade in a lookback window, attribute realized PnL to
the contributing models for that run (uniformly across models in per_model_json),
then aggregate by (regime, horizon, model) and update `model_trust_regime` with
EMA smoothing.
"""

import os
from collections import defaultdict
from typing import Dict, Tuple

from loguru import logger

from ..infra.db import get_conn, upsert_model_trust_regime


def run(window_days: int = None, beta: float = None) -> Dict[Tuple[str, str], Dict[str, float]]:
    window_days = window_days or int(os.getenv("PAPER_ADAPT_WINDOW_DAYS", "30"))
    beta = beta or float(os.getenv("PAPER_ADAPT_BETA", "0.1"))
    alpha_min = float(os.getenv("REGIME_ALPHA_MIN", "0.7"))
    alpha_max = float(os.getenv("REGIME_ALPHA_MAX", "1.3"))

    sql = (
        """
        SELECT pnl.realized_pnl, r.label, pr.horizon, pr.per_model_json
        FROM paper_pnl pnl
        JOIN paper_positions p USING (pos_id)
        JOIN trades_suggestions t ON (t.run_id = p.meta_json->>'run_id')
        JOIN predictions pr ON (pr.run_id = t.run_id)
        LEFT JOIN regimes r ON (r.run_id = t.run_id)
        WHERE p.status='CLOSED' AND p.opened_at >= now() - (%s || ' days')::interval
        """
    )
    buckets: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (str(window_days),))
            for pnl, reg, hz, per_model in cur.fetchall() or []:
                try:
                    reg = str(reg or "range")
                    hz = str(hz or "4h")
                    pm: Dict[str, dict] = per_model or {}
                    if not pm:
                        continue
                    mlist = list(pm.keys())
                    if not mlist:
                        continue
                    share = float(pnl or 0.0) / float(len(mlist))
                    for m in mlist:
                        buckets[(reg, hz)][str(m)] += share
                except Exception:
                    continue

    # Normalize and apply EMA update around 1.0 baseline
    result: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (reg, hz), msum in buckets.items():
        if not msum:
            continue
        # scale to [-1..1] by max abs
        mx = max(abs(v) for v in msum.values()) or 1.0
        updates = {m: (v / mx) for m, v in msum.items()}  # -1..1
        new_w: Dict[str, float] = {}
        for m, u in updates.items():
            base = 1.0 + 0.2 * float(u)  # map to ~[0.8..1.2]
            # fetch prev via select; for simplicity rely on upsert with smoothing on caller side
            # compute smoothed alpha and clip
            try:
                a = base * beta + (1.0) * (1.0 - beta)
                a = max(alpha_min, min(alpha_max, a))
                upsert_model_trust_regime(reg, hz, m, float(a))
                new_w[m] = float(a)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"adapt upsert failed: reg={reg} hz={hz} m={m}: {e}")
        result[(reg, hz)] = new_w
    logger.info(f"paper_adapt: updated {sum(len(v) for v in result.values())} weights across {len(result)} buckets")
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()

