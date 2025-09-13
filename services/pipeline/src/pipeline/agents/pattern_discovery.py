from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import time

import pandas as pd

from ..infra.db import upsert_technical_pattern, insert_pattern_metrics
from ..reasoning.llm import call_openai_json


@dataclass
class DiscoveryInput:
    symbol: str = "BTC/USDT"
    provider: str = "binance"
    timeframe: str = "1h"  # 1h to keep it tractable
    days: int = 120  # lookback days
    move_threshold: float = 0.05  # 5% 24h move
    window_hours: int = 24  # event window size
    lookback_hours_pre: int = 48  # pattern search window before event
    sample_limit: int = 40  # cap number of events analyzed
    dry_run: bool = True  # if True, do not write to DB


def _fetch_ohlcv(symbol: str, provider: str, timeframe: str, since_ms: int) -> List[List[float]]:
    import ccxt  # type: ignore

    ex = getattr(ccxt, provider)({"enableRateLimit": True, "timeout": 20000})
    out: List[List[float]] = []
    limit = 1000
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        out.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < limit:
            break
        # be gentle with rate limits
        time.sleep(0.2)
    return out


def _find_events(df: pd.DataFrame, window_h: int, thr: float) -> List[int]:
    # Index of starting bar for which close[t+window] deviates > thr
    step = max(1, int(window_h))
    events: List[int] = []
    for i in range(0, len(df) - step):
        p0 = float(df["close"].iloc[i])
        p1 = float(df["close"].iloc[i + step])
        if p0 <= 0:
            continue
        ch = abs(p1 - p0) / p0
        if ch >= thr:
            events.append(i)
    return events


def _extract_pre_window(df: pd.DataFrame, idx: int, lookback_h: int) -> pd.DataFrame:
    L = max(1, int(lookback_h))
    lo = max(0, idx - L)
    hi = idx
    return df.iloc[lo:hi].copy()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _summarize_window(win: pd.DataFrame) -> Dict[str, Any]:
    # Simple summary stats useful for LLM to generalize
    out: Dict[str, Any] = {}
    try:
        close = win["close"].astype(float)
        ret = close.pct_change().fillna(0.0)
        out["ret_sum"] = float(ret.sum())
        out["ret_std"] = float(ret.std(ddof=0)) if len(ret) else 0.0
        try:
            out["ret_skew"] = float(ret.skew()) if len(ret) else 0.0
        except Exception:
            out["ret_skew"] = 0.0
        # RSI / MACD / BB width
        rsi = _rsi(close)
        macd, macd_signal, macd_hist = _macd(close)
        out["rsi_mean"] = float(rsi.mean()) if len(rsi) else 50.0
        out["rsi_last"] = float(rsi.iloc[-1]) if len(rsi) else 50.0
        out["macd_hist_mean"] = float(macd_hist.mean()) if len(macd_hist) else 0.0
        atr_mean = float(win.get("atr_14", pd.Series()).mean() or 0.0)
        bbw_mean = float(win.get("bb_width", pd.Series()).mean() or 0.0)
        out["atr_mean"] = atr_mean
        out["bb_width_mean"] = bbw_mean
        # Ratios
        try:
            out["rsi_last_over_mean"] = float(out["rsi_last"]) / max(1e-6, out["rsi_mean"]) if out["rsi_mean"] else 1.0
        except Exception:
            out["rsi_last_over_mean"] = 1.0
        try:
            out["bb_over_atr"] = float(bbw_mean) / max(1e-6, atr_mean) if atr_mean else 0.0
        except Exception:
            out["bb_over_atr"] = 0.0
        # pattern counts
        for c in win.columns:
            if str(c).startswith("pat_"):
                try:
                    out[f"count_{c}"] = int(win[c].astype(float).sum())
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _propose_patterns(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sys = (
        "You are a market pattern discovery researcher. You are given a list of summaries of contexts "
        "preceding large moves. Find 1-3 recurring, formalizable patterns with a concise description and a JSON definition. "
        "Definition MUST include selection_rules as a list of {key, op, value}, where key references available sample fields "
        "(e.g., count_pat_*, rsi_mean, rsi_last, macd_hist_mean, atr_mean, bb_width_mean), op in ['>','>=','<','<=','==','!='], value is numeric. "
        "Return strictly JSON {\"patterns\":[{name,category,timeframe,description,expected_direction,confidence_default,definition_json:{selection_rules:[...]}}]}."
    )
    usr = f"Samples: {samples[:50]}"
    data = call_openai_json(sys, usr, model=None, temperature=0.3)
    pats = (data or {}).get("patterns", []) or []
    # normalize items
    out = []
    for p in pats[:3]:
        try:
            out.append(
                {
                    "name": str(p.get("name")),
                    "category": str(p.get("category", "chart")),
                    "timeframe": str(p.get("timeframe", "1h")),
                    "definition_json": p.get("definition_json") or {},
                    "description": str(p.get("description", "")),
                    "expected_direction": str(p.get("expected_direction", "up")),
                    "confidence_default": float(p.get("confidence_default", 0.6) or 0.6),
                }
            )
        except Exception:
            continue
    return out


def run(inp: DiscoveryInput) -> Dict[str, Any]:
    # Fetch OHLCV
    since_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=int(inp.days))).timestamp() * 1000)
    raw = _fetch_ohlcv(inp.symbol, inp.provider, inp.timeframe, since_ms)
    if not raw or len(raw) < 100:
        return {"status": "no-data"}
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])  # type: ignore
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    # Events and samples
    events = _find_events(df, inp.window_hours, inp.move_threshold)
    events = events[: max(1, int(inp.sample_limit))]
    if not events:
        return {"status": "no-events"}

    samples: List[Dict[str, Any]] = []
    for i in events:
        win = _extract_pre_window(df, i, inp.lookback_hours_pre)
        s = _summarize_window(win)
        # future direction over window_hours
        try:
            p0 = float(df["close"].iloc[i])
            p1 = float(df["close"].iloc[min(len(df)-1, i + int(inp.window_hours))])
            s["future_change"] = (p1 - p0) / max(1e-9, p0)
            s["future_dir"] = 1 if s["future_change"] > 0 else (-1 if s["future_change"] < 0 else 0)
        except Exception:
            s["future_change"], s["future_dir"] = 0.0, 0
        samples.append(s)

    patterns = _propose_patterns(samples)

    def _match(sample: Dict[str, Any], rules: List[Dict[str, Any]]) -> bool:
        ops = {
            '>': lambda a,b: a > b,
            '>=': lambda a,b: a >= b,
            '<': lambda a,b: a < b,
            '<=': lambda a,b: a <= b,
            '==': lambda a,b: a == b,
            '!=': lambda a,b: a != b,
        }
        for r in rules or []:
            k = str(r.get('key'))
            op = str(r.get('op'))
            v = float(r.get('value', 0))
            if k not in sample or op not in ops:
                return False
            try:
                if not ops[op](float(sample[k]), v):
                    return False
            except Exception:
                return False
        return True

    def _binom_p_value(success: int, n: int, p0: float = 0.5) -> float:
        import math
        if n <= 0:
            return 1.0
        # one-sided p-value (success rate >= p0)
        def comb(n,k):
            from math import comb as _c
            return _c(n,k)
        pv = 0.0
        for k in range(success, n+1):
            pv += comb(n,k) * (p0**k) * ((1-p0)**(n-k))
        return min(1.0, max(0.0, pv))

    # Acceptance thresholds (configurable via agent_configurations)
    try:
        from ..infra.db import fetch_agent_config as _fetch_cfg
        cfg = _fetch_cfg("PatternDiscoveryAgent") or {}
        params = cfg.get("parameters") or {}
        thr_min_matches = int(params.get("min_match_count", 8))
        thr_min_rate = float(params.get("min_success_rate", 0.65))
        thr_max_p = float(params.get("max_p_value", 0.05))
    except Exception:
        thr_min_matches, thr_min_rate, thr_max_p = 8, 0.65, 0.05

    results = []
    written = 0
    for p in patterns:
        rules = (p.get('definition_json') or {}).get('selection_rules') or []
        matched = [s for s in samples if _match(s, rules)]
        m = len(matched)
        succ = 0
        dir_expect = 1 if str(p.get('expected_direction','up')).lower() == 'up' else -1
        for s in matched:
            if int(s.get('future_dir', 0)) == dir_expect:
                succ += 1
        rate = (succ / m) if m > 0 else 0.0
        pval = _binom_p_value(succ, m, 0.5)
        # store metrics
        try:
            insert_pattern_metrics(
                symbol=inp.symbol,
                timeframe=inp.timeframe,
                window_hours=int(inp.window_hours),
                move_threshold=float(inp.move_threshold),
                sample_count=len(samples),
                pattern_name=p['name'],
                expected_direction=('up' if dir_expect==1 else 'down'),
                match_count=m,
                success_count=succ,
                success_rate=rate,
                p_value=pval,
                definition=p.get('definition_json') or {},
                summary={"rules": rules},
            )
        except Exception:
            pass
        # upsert to technical_patterns if strong and not dry_run
        if (not inp.dry_run) and m >= thr_min_matches and rate >= thr_min_rate and pval <= thr_max_p:
            try:
                upsert_technical_pattern(
                    name=p["name"],
                    category=p.get("category","chart"),
                    timeframe=p.get("timeframe", inp.timeframe),
                    definition=p.get("definition_json") or {},
                    description=p.get("description"),
                    source="discover",
                    confidence_default=float(p.get("confidence_default", 0.6) or 0.6),
                )
                written += 1
            except Exception:
                pass
        results.append({"pattern": p, "match_count": m, "success": succ, "success_rate": rate, "p_value": pval})

    return {"status": "ok", "events": len(events), "patterns": results, "written": written}


def main():
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.agents.pattern_discovery '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = DiscoveryInput(**json.loads(sys.argv[1]))
    res = run(payload)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
