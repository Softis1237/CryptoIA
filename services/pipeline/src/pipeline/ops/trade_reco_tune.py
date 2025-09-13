from __future__ import annotations

"""Heuristic tuner for TradeRecommend parameters based on paper trades.

Calculates SL/TP hit rates and suggests updated rr_target and k_atr.
Stores suggestion as a new version in agent_configurations for agent_name=TradeRecommend.
"""

import os
from typing import Tuple

import numpy as np
from loguru import logger

from ..infra.db import get_conn, insert_agent_config


def _fetch_closed_positions(window_days: int = 30) -> list[tuple[float, float, float, str]]:
    sql = (
        """
        SELECT p.entry, p.sl, p.tp, t.reason
        FROM paper_trades t
        JOIN paper_positions p USING (pos_id)
        WHERE t.side='CLOSE' AND p.opened_at >= now() - (%s || ' days')::interval
        """
    )
    out: list[tuple[float, float, float, str]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (str(window_days),))
            for entry, sl, tp, reason in cur.fetchall() or []:
                try:
                    out.append((float(entry or 0.0), float(sl or 0.0), float(tp or 0.0), str(reason or "")))
                except Exception:
                    continue
    return out


def _suggest_params(data: list[tuple[float, float, float, str]]) -> Tuple[float, float]:
    if not data:
        return 1.6, 1.5
    hits_tp = sum(1 for _, _, _, r in data if str(r) == "take_profit")
    hits_sl = sum(1 for _, _, _, r in data if str(r) == "stop_loss")
    total = max(1, len(data))
    tp_rate = hits_tp / total
    sl_rate = hits_sl / total
    # Simple heuristic: if TP too rare (<30%), lower rr_target a bit; if too frequent (>60%), raise
    rr = 1.6 + (0.3 if tp_rate > 0.6 else (-0.2 if tp_rate < 0.3 else 0.0))
    rr = float(min(2.2, max(1.0, rr)))
    # k_atr: if SL hits >> TP, shrink SL distance slightly
    katr = 1.5 + (-0.2 if sl_rate > 0.6 else (0.2 if sl_rate < 0.3 else 0.0))
    katr = float(min(2.5, max(0.8, katr)))
    return rr, katr


def run(window_days: int = None) -> dict:
    window_days = window_days or int(os.getenv("TR_TUNE_WINDOW_DAYS", "30"))
    rows = _fetch_closed_positions(window_days)
    rr, katr = _suggest_params(rows)
    ver = insert_agent_config(
        agent_name="TradeRecommend",
        system_prompt=None,
        parameters={"rr_target": rr, "k_atr": katr},
        make_active=True,
    )
    logger.info(f"TradeRecommend tuned: rr_target={rr:.2f} k_atr={katr:.2f} (version {ver})")
    return {"rr_target": rr, "k_atr": katr, "version": ver}


def main() -> None:
    run()


if __name__ == "__main__":
    main()

