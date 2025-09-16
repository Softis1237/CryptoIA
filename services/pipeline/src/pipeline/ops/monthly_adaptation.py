from __future__ import annotations

"""Monthly adaptation job: analyze pattern success, update weights, insert lessons.

This is a lightweight scaffold to be scheduled monthly (Windmill/Cron/Airflow).
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..infra.db import get_conn, insert_agent_lesson, insert_agent_config


@dataclass
class MonthlyInput:
    lookback_days: int = 30


def _aggregate_patterns(days: int) -> Dict[str, Any]:
    sql = (
        "SELECT pattern_name, SUM(match_count) AS m, SUM(success_count) AS s "
        "FROM pattern_discovery_metrics WHERE created_at >= now() - interval '%s days' "
        "GROUP BY pattern_name ORDER BY s::float/NULLIF(m,0) DESC NULLS LAST LIMIT 20"
    )
    out: Dict[str, Any] = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, (int(days),))
                for name, m, s in cur.fetchall() or []:
                    out[str(name)] = {"matches": int(m or 0), "wins": int(s or 0), "winrate": (float(s or 0) / max(1, int(m or 0)))}
            except Exception:
                pass
    return out


def run(inp: MonthlyInput) -> Dict[str, Any]:
    days = int(inp.lookback_days)
    pats = _aggregate_patterns(days)

    # Aggregate strategic verdicts
    def _agg_strategic() -> Dict[str, Any]:
        sql = (
            "SELECT agent_name, verdict, COUNT(1) FROM strategic_verdicts "
            "WHERE ts >= now() - interval '%s days' GROUP BY agent_name, verdict"
        )
        out: Dict[str, Any] = {}
        with get_conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql, (int(days),))
                    for agent, verdict, cnt in cur.fetchall() or []:
                        key = f"{agent}:{verdict}"
                        out[key] = int(cnt or 0)
                except Exception:
                    pass
        return out

    strat = _agg_strategic()

    # Adjust ConfidenceAggregator weights (pattern/smc/whale) heuristically
    new_params: Dict[str, float] = {}
    try:
        top = sorted(pats.items(), key=lambda kv: kv[1].get("winrate", 0.0), reverse=True)[:5]
        if top:
            # modestly raise w_pattern if good patterns exist
            new_params["w_pattern"] = 0.12
    except Exception:
        pass
    try:
        smc_bull = strat.get("SMC Analyst:SMC_BULLISH_SETUP", 0)
        smc_bear = strat.get("SMC Analyst:SMC_BEARISH_SETUP", 0)
        whale_bull = strat.get("Whale Watcher:WHALE_BULLISH", 0)
        whale_bear = strat.get("Whale Watcher:WHALE_BEARISH", 0)
        if smc_bull + smc_bear >= 10:
            new_params["w_smc"] = 0.12
        if whale_bull + whale_bear >= 10:
            new_params["w_whale"] = 0.10
    except Exception:
        pass
    version = None
    try:
        if new_params:
            version = insert_agent_config("ConfidenceAggregator", None, new_params, make_active=True)
    except Exception:
        version = None

    # Analyze winrate by regimes and adapt AlertPriority presets (trend vs range)
    def _regime_stats(days: int) -> Dict[str, float]:
        sql = (
            "SELECT regime_label, AVG(CASE WHEN direction_correct THEN 1 ELSE 0 END)::float AS winrate, COUNT(1) AS n "
            "FROM prediction_outcomes WHERE created_at >= now() - interval '%s days' AND regime_label IS NOT NULL GROUP BY regime_label"
        )
        out: Dict[str, float] = {}
        with get_conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql, (int(days),))
                    rows = cur.fetchall() or []
                    for regime_label, winrate, n in rows:
                        out[str(regime_label)] = float(winrate or 0.0)
                except Exception:
                    pass
        return out

    rstats = _regime_stats(days)
    try:
        # Heuristic: if trend_up/trend_down winrates dominate → trend preset; otherwise → range preset
        tr = (rstats.get("trend_up", 0.0) + rstats.get("trend_down", 0.0)) / 2.0
        rg = rstats.get("range", 0.0)
        ap_params = None
        if tr >= rg:
            # Trend preset: усилить MOMENTUM/ATR/NEWS; снизить L2_WALL
            ap_params = {
                "weights": {"MOMENTUM": 1.4, "ATR_SPIKE": 1.3, "NEWS": 1.5, "L2_WALL": 0.7},
                "thresholds": {"low": 0.0, "medium": 1.0, "high": 2.0, "critical": 2.8},
                "critical_combos": [["MOMENTUM", "VOL_SPIKE"], ["MOMENTUM", "PATTERN_BULL_ENGULF"], ["MOMENTUM", "PATTERN_BEAR_ENGULF"]],
            }
        else:
            # Range preset: усилить L2_IMBALANCE/паттерны, чуть снизить критический порог
            ap_params = {
                "weights": {"L2_IMBALANCE": 1.2, "PATTERN_BULL_ENGULF": 1.3, "PATTERN_BEAR_ENGULF": 1.3, "MOMENTUM": 1.1},
                "thresholds": {"low": 0.0, "medium": 1.0, "high": 2.2, "critical": 3.2},
                "critical_combos": [["L2_IMBALANCE", "VOL_SPIKE"], ["PATTERN_BULL_ENGULF", "VOL_SPIKE"], ["PATTERN_BEAR_ENGULF", "VOL_SPIKE"]],
            }
        if ap_params:
            insert_agent_config("AlertPriority", None, ap_params, make_active=True)
    except Exception:
        pass

    # Lessons (more detailed)
    try:
        meta = {"patterns": pats, "strategic": strat, "updated_params": new_params}
        insert_agent_lesson(
            "Ежемесячный анализ: обновлены веса ConfidenceAggregator по паттернам и стратегическим агентам.",
            scope="global",
            meta=meta,
        )
    except Exception:
        pass
    return {"status": "ok", "patterns": pats, "strategic": strat, "version": version, "regime_stats": rstats}


def main() -> None:
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.ops.monthly_adaptation '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = MonthlyInput(**json.loads(sys.argv[1]))
    out = run(payload)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
