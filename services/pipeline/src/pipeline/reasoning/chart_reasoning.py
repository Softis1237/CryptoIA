from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..infra.s3 import download_bytes
from .llm import call_openai_json
from ..infra.db import fetch_technical_patterns, fetch_agent_config


@dataclass
class ChartReasoningInput:
    features_path_s3: str
    top_k_patterns: int = 10  # how many non-zero pattern flags to surface


def _load_df(features_s3: str) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = download_bytes(features_s3)
    df = pq.read_table(pa.BufferReader(raw)).to_pandas().sort_values("ts")
    return df


def _extract_snapshot(last: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Core price/volatility context
    for k in [
        "ts",
        "close",
        "atr_14",
        "ret_1m",
        "macd",
        "macd_signal",
        "bb_width",
        "vwap",
        "volume_z",
        "regime_trend_up",
        "regime_trend_down",
        "regime_range",
        "regime_volatility",
        "regime_crash",
        "regime_healthy_rise",
    ]:
        if k in last.index:
            try:
                out[k] = float(last[k])
            except Exception:
                out[k] = last[k]
    # Pattern flags
    pats: List[str] = []
    for c in last.index:
        if str(c).startswith("pat_"):
            try:
                v = float(last[c])
                if v and v != 0.0:
                    pats.append(c)
            except Exception:
                continue
    out["patterns_present"] = sorted(pats)[:50]
    return out


def run(inp: ChartReasoningInput) -> Dict[str, Any]:
    df = _load_df(inp.features_path_s3)
    last = df.iloc[-1]
    snap = _extract_snapshot(last)
    # Add local levels/ATR corridor for richer context
    try:
        tail = df.tail(120)
        close = tail["close"].astype(float)
        snap["local_high_50"] = float(close.tail(50).max()) if len(close) >= 10 else float(last.get("close", 0.0))
        snap["local_low_50"] = float(close.tail(50).min()) if len(close) >= 10 else float(last.get("close", 0.0))
        atr = float(last.get("atr_14", 0.0) or 0.0)
        cval = float(last.get("close", 0.0) or 0.0)
        snap["atr_band_lower"] = cval - atr
        snap["atr_band_upper"] = cval + atr
        # Quantiles of last 120 bars
        if len(close) >= 20:
            snap["q10"] = float(close.quantile(0.10))
            snap["q50"] = float(close.quantile(0.50))
            snap["q90"] = float(close.quantile(0.90))
        # Classic pivots from last 24h window if available
        w = df.tail(min(len(df), 1440))
        if len(w) >= 10:
            H = float(w["high"].astype(float).max())
            L = float(w["low"].astype(float).min())
            C = float(w["close"].astype(float).iloc[-1])
            P = (H + L + C) / 3.0
            R1 = 2 * P - L
            S1 = 2 * P - H
            R2 = P + (H - L)
            S2 = P - (H - L)
            R3 = R1 + (H - L)
            S3 = S1 - (H - L)
            snap["pivot_P"] = P
            snap["pivot_R1"] = R1
            snap["pivot_S1"] = S1
            snap["pivot_R2"] = R2
            snap["pivot_S2"] = S2
            snap["pivot_R3"] = R3
            snap["pivot_S3"] = S3
        # VWAP and median levels (if available)
        try:
            if "vwap" in df.columns:
                snap["vwap_last"] = float(df["vwap"].astype(float).iloc[-1])
        except Exception:
            pass
        snap["median_120"] = float(close.median()) if len(close) > 0 else cval
        # Distances from price to reference levels (percent)
        def _dist(a: float, b: float) -> float:
            return ((a - b) / b) if b else 0.0
        try:
            snap["dist_to_vwap_pct"] = float(_dist(cval, float(snap.get("vwap_last", 0.0) or 0.0)))
            snap["dist_to_median_pct"] = float(_dist(cval, float(snap.get("median_120", 0.0) or 0.0)))
            for key in ["pivot_P","pivot_R1","pivot_S1","pivot_R2","pivot_S2","pivot_R3","pivot_S3"]:
                if key in snap:
                    snap[f"dist_to_{key}_pct"] = float(_dist(cval, float(snap.get(key) or 0.0)))
        except Exception:
            pass
    except Exception:
        pass
    all_defs = fetch_technical_patterns()
    defs_map = {p.get("name"): (p.get("description") or "") for p in all_defs}
    # Reduce pattern descriptions to those present (strip pat_ prefix where applicable)
    present = []
    for p in snap.get("patterns_present", []):
        name = str(p).replace("pat_", "")
        present.append({"name": name, "meaning": defs_map.get(name, "")})
    # Compose prompts
    default_sys = (
        "You are a senior technical analyst. Think step-by-step, but return only JSON with keys: "
        "technical_sentiment (bullish|bearish|neutral), confidence_score (0..1), key_observations (array of concise bullets). "
        "Use the provided indicators and detected patterns (with their typical meaning) to justify your view."
    )
    cfg = fetch_agent_config("ChartReasoningAgent") or {}
    sys_prompt = cfg.get("system_prompt") or default_sys
    params = cfg.get("parameters") or {}
    model = params.get("model") if isinstance(params, dict) else None
    temperature = float(params.get("temperature", 0.2)) if isinstance(params, dict) else 0.2
    usr = (f"Snapshot: {snap}\n" f"Patterns (meaning): {present}")
    data = call_openai_json(sys_prompt, usr, model=model, temperature=temperature)
    # Minimal validation
    out = {
        "technical_sentiment": str((data or {}).get("technical_sentiment", "neutral")),
        "confidence_score": float((data or {}).get("confidence_score", 0.5) or 0.5),
        "key_observations": (data or {}).get("key_observations", []) or [],
    }
    # Coerce observations into list[str]
    out["key_observations"] = [str(x) for x in out["key_observations"]][:8]
    # Clamp confidence
    try:
        out["confidence_score"] = max(0.0, min(1.0, float(out["confidence_score"])) )
    except Exception:
        out["confidence_score"] = 0.5
    return out


def main():
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.reasoning.chart_reasoning '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = ChartReasoningInput(**json.loads(sys.argv[1]))
    res = run(payload)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
