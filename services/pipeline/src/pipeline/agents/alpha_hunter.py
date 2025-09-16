from __future__ import annotations

import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..infra.db import (
    insert_elite_trades,
    insert_alpha_snapshot,
    upsert_alpha_strategy,
    fetch_active_alpha_strategies,
)
from ..reasoning.llm import call_openai_json
from ..data.ingest_prices import IngestPricesInput, run as run_prices
from ..features.features_calc import FeaturesCalcInput, run as run_features
from ..infra.metrics import push_values
from ..infra.s3 import download_bytes

try:  # optional heavy deps (numpy/pyarrow) used in backtests
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # noqa: BLE001
    pa = None
    pq = None


Numeric = (int, float)


_HTTP_SESSION: requests.Session | None = None


def _http_session() -> requests.Session:
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        session = requests.Session()
        retry = Retry(
            total=int(os.getenv("ALPHA_HTTP_RETRIES", "3")),
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=float(os.getenv("ALPHA_HTTP_BACKOFF", "0.6")),
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": os.getenv(
                    "ALPHA_HTTP_UA",
                    "AlphaHunterBot/1.0 (+https://alpha)",
                )
            }
        )
        _HTTP_SESSION = session
    return _HTTP_SESSION


@dataclass
class CollectInput:
    source: str = "binance"
    sources: List[str] = None  # type: ignore[assignment]
    symbol: str = "BTC/USDT"
    top_n: int = 10
    lookback_days: int = 7
    dry_run: bool = True
    source_url: str | None = None  # optional HTTP endpoint returning leaderboard trades JSON


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    arr = sorted(values)
    pos = (len(arr) - 1) * (q / 100.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(arr[int(pos)])
    weight = pos - lower
    return float(arr[lower] * (1 - weight) + arr[upper] * weight)


def _fetch_from_json(url: str, top_n: int, default_source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        resp = _http_session().get(url, timeout=float(os.getenv("ALPHA_HTTP_TIMEOUT", "15")))
        resp.raise_for_status()
        data = resp.json() or []
        for it in data[: int(top_n)]:
            try:
                rows.append(
                    {
                        "source": str(it.get("source") or default_source),
                        "trader_id": str(it.get("trader_id") or it.get("uid") or "unknown"),
                        "symbol": str(it.get("symbol") or it.get("pair") or "BTC/USDT"),
                        "side": str(it.get("side") or it.get("direction") or "LONG").upper(),
                        "entry_price": float(it.get("entry_price")) if it.get("entry_price") not in {None, ""} else None,
                        "ts": datetime.fromisoformat(str(it.get("ts") or it.get("timestamp")).replace("Z", "+00:00")),
                        "pnl": float(it.get("pnl", it.get("roi", 0.0)) or 0.0),
                        "meta": it.get("meta") or {},
                    }
                )
            except Exception:
                continue
    except Exception as e:  # noqa: BLE001
        logger.debug(f"alpha_hunter remote fetch failed for {url}: {e}")
    return rows


def _binance_public_fetch(top_n: int, symbol: str) -> List[Dict[str, Any]]:
    url = os.getenv("ALPHA_BINANCE_API")
    if url:
        return _fetch_from_json(url, top_n, "binance")
    # Best-effort scrape of Binance leaderboard public API (positions, not private data)
    try:
        session = _http_session()
        rank_url = "https://www.binance.com/bapi/futures/v1/public/future/leaderboard/getLeaderboardRank"
        resp = session.get(rank_url, timeout=float(os.getenv("ALPHA_HTTP_TIMEOUT", "15")))
        resp.raise_for_status()
        payload = resp.json() or {}
        entries = []
        for item in (payload.get("data") or [])[: int(top_n)]:
            enc_uid = item.get("encryptedUid")
            if not enc_uid:
                continue
            try:
                pos_url = "https://www.binance.com/bapi/futures/v1/public/future/leaderboard/getOtherPosition"
                p_resp = session.get(
                    pos_url,
                    params={"encryptedUid": enc_uid},
                    timeout=float(os.getenv("ALPHA_HTTP_TIMEOUT", "15")),
                )
                p_resp.raise_for_status()
                pdata = p_resp.json() or {}
                positions = pdata.get("data", {}).get("otherPositionRetList") or []
                for pos in positions:
                    pair = str(pos.get("symbol") or symbol)
                    if symbol.replace("/", "") not in pair:
                        continue
                    direction = "LONG" if float(pos.get("amount", 0.0)) >= 0 else "SHORT"
                    open_time = datetime.fromtimestamp(pos.get("entryTime") / 1000.0, tz=timezone.utc)
                    entries.append(
                        {
                            "source": "binance",
                            "trader_id": str(item.get("nickName") or enc_uid),
                            "symbol": pair if "/" in pair else f"{pair[:-4]}/{pair[-4:]}",
                            "side": direction,
                            "entry_price": float(pos.get("entryPrice")) if pos.get("entryPrice") else None,
                            "ts": open_time,
                            "pnl": float(pos.get("roe", 0.0)) / 100.0,
                            "meta": {
                                "nickname": item.get("nickName"),
                                "roi": pos.get("roe"),
                                "rank": item.get("rank"),
                            },
                        }
                    )
            except Exception:
                continue
        entries.sort(key=lambda r: abs(r.get("pnl", 0.0)), reverse=True)
        return entries[: int(top_n)]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"alpha_hunter binance scrape failed: {e}")
        return []


def _bybit_public_fetch(top_n: int, symbol: str) -> List[Dict[str, Any]]:
    url = os.getenv("ALPHA_BYBIT_API")
    if url:
        return _fetch_from_json(url, top_n, "bybit")
    try:
        session = _http_session()
        leaderboard_url = "https://api2.bybit.com/contract/v5/public/leaderboard/leader/getList"
        params = {"periodType": "DAILY", "pageSize": max(10, top_n)}
        resp = session.get(
            leaderboard_url,
            params=params,
            timeout=float(os.getenv("ALPHA_HTTP_TIMEOUT", "15")),
        )
        resp.raise_for_status()
        data = resp.json() or {}
        rows: List[Dict[str, Any]] = []
        for item in (data.get("result", {}).get("list") or [])[: int(top_n)]:
            uid = item.get("uid")
            if not uid:
                continue
            try:
                detail_url = "https://api2.bybit.com/contract/v5/public/leaderboard/user/list"
                d_resp = session.get(
                    detail_url,
                    params={"uid": uid},
                    timeout=float(os.getenv("ALPHA_HTTP_TIMEOUT", "15")),
                )
                d_resp.raise_for_status()
                d_data = d_resp.json() or {}
                trades = d_data.get("result", {}).get("list") or []
                for tr in trades:
                    pair = str(tr.get("symbol") or symbol)
                    if symbol.replace("/", "") not in pair.replace("-", ""):
                        continue
                    ts_val = tr.get("updatedAt") or tr.get("createdAt")
                    if isinstance(ts_val, (int, float)):
                        ts_dt = datetime.fromtimestamp(float(ts_val) / 1000.0, tz=timezone.utc)
                    else:
                        ts_dt = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
                    rows.append(
                        {
                            "source": "bybit",
                            "trader_id": str(uid),
                            "symbol": pair,
                            "side": str(tr.get("side") or "long").upper(),
                            "entry_price": float(tr.get("entryPrice")) if tr.get("entryPrice") else None,
                            "ts": ts_dt,
                            "pnl": float(tr.get("pnl", 0.0)) / 100.0,
                            "meta": {
                                "nickname": item.get("nickname"),
                                "roi": tr.get("pnl"),
                            },
                        }
                    )
            except Exception:
                continue
        rows.sort(key=lambda r: abs(r.get("pnl", 0.0)), reverse=True)
        return rows[: int(top_n)]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"alpha_hunter bybit fetch failed: {e}")
        return []


def _fetch_leaderboard_trades(inp: CollectInput) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    sources = inp.sources or [inp.source]
    # First, user-provided endpoint overrides
    if inp.source_url or os.getenv("ALPHA_LEADERBOARD_URL"):
        url = inp.source_url or os.getenv("ALPHA_LEADERBOARD_URL")
        rows = _fetch_from_json(url, inp.top_n, sources[0])
        if rows:
            candidates.extend(rows)
    for src in sources:
        src = src.lower()
        if src == "binance":
            candidates.extend(_binance_public_fetch(inp.top_n, inp.symbol))
        elif src == "bybit":
            candidates.extend(_bybit_public_fetch(inp.top_n, inp.symbol))
    if not candidates:
        now = datetime.now(timezone.utc)
        for i in range(min(3, int(inp.top_n))):
            ts = now - timedelta(days=min(inp.lookback_days, 7), hours=(i + 1) * 6)
            candidates.append(
                {
                    "source": sources[0] if sources else inp.source,
                    "trader_id": f"TRADER_{i+1:02d}",
                    "symbol": inp.symbol,
                    "side": "LONG" if i % 2 == 0 else "SHORT",
                    "entry_price": None,
                    "ts": ts,
                    "pnl": 0.15 if i % 2 == 0 else 0.1,
                    "meta": {"synthetic": True},
                }
            )
    candidates.sort(key=lambda r: abs(r.get("pnl", 0.0)), reverse=True)
    return candidates[: int(inp.top_n)]


def _load_feature_vector(features_path_s3: str) -> Tuple[Dict[str, Any], Optional[Any]]:
    if not features_path_s3:
        return {}, None
    if pq is None or pa is None:
        return {}, None
    try:
        raw = download_bytes(features_path_s3)
        table = pq.read_table(pa.BufferReader(raw))
        df = table.to_pandas().sort_values("ts")
        if df.empty:
            return {}, df
        vec = {}
        last = df.iloc[-1]
        for k, v in last.items():
            if isinstance(v, Numeric):
                try:
                    vec[k] = float(v)
                except Exception:
                    continue
        return vec, df
    except Exception as e:  # noqa: BLE001
        logger.debug(f"alpha_hunter feature vector load failed: {e}")
        return {}, None


def _snapshot_context(symbol: str, trade_ts: datetime) -> Tuple[Dict[str, Any], Optional[Any]]:
    end_ts = int(trade_ts.replace(tzinfo=timezone.utc).timestamp())
    start_ts = end_ts - 6 * 3600
    prices = run_prices(
        IngestPricesInput(
            run_id=f"alpha_ctx_{end_ts}",
            slot="alpha",
            symbols=[symbol.replace("/", "")],
            start_ts=start_ts,
            end_ts=end_ts,
            provider="binance",
            timeframe="1m",
        )
    )
    feats = run_features(
        FeaturesCalcInput(
            prices_path_s3=getattr(prices, "prices_path_s3", ""),
            run_id=f"alpha_ctx_{end_ts}",
            slot="alpha",
        )
    )
    features_path = getattr(feats, "features_path_s3", "")
    feature_vector, df = _load_feature_vector(features_path)
    ctx_json = {
        "features_path_s3": features_path,
        "feature_schema": getattr(feats, "feature_schema", []),
        "snapshot_ts": getattr(feats, "snapshot_ts", ""),
        "feature_vector": feature_vector,
    }
    return ctx_json, df


def _propose_strategies(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sys = (
        "You are Alpha Hunter. Given snapshots of market context around profitable elite trades, "
        "find 1-3 recurring formalizable strategies. Each strategy must have selection_rules "
        "as a list of {key, op in ['>','>=','<','<=','==','!='], value (numeric)} operating on available features "
        "like rsi_14, macd_hist, bb_width, volume_z, news_ctx_score, regime_*, and optionally agents signals. "
        "Return JSON {\"strategies\":[{name, description, definition_json:{selection_rules:[...]}, required_agents?: [\"Whale Watcher\", \"SMC Analyst\"]}]}"
    )
    usr = f"Samples: {samples[:50]}"
    data = call_openai_json(sys, usr, model=None, temperature=0.2)
    return (data or {}).get("strategies", []) or []


def run_collect_and_mine(inp: CollectInput) -> Dict[str, Any]:
    if inp.sources is None:
        inp.sources = [inp.source]
    trades = _fetch_leaderboard_trades(inp)
    inserted = 0
    if not inp.dry_run and trades:
        try:
            inserted = insert_elite_trades(trades)
        except Exception:
            inserted = 0
    contexts: List[Dict[str, Any]] = []
    for t in trades:
        try:
            ctx_json, df = _snapshot_context(t["symbol"], t["ts"])  # type: ignore[index]
            if not inp.dry_run:
                try:
                    insert_alpha_snapshot(
                        t.get("id") or "00000000-0000-0000-0000-000000000000",
                        {k: v for k, v in ctx_json.items() if k != "feature_vector"},
                    )
                except Exception:
                    pass
            contexts.append({"trade": t, "ctx": ctx_json, "df": df})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"alpha snapshot failed: {e}")
            continue
    samples: List[Dict[str, Any]] = []
    for it in contexts:
        s = {k: it["trade"].get(k) for k in ["symbol", "side", "pnl", "source", "meta"]}
        ctx = it.get("ctx", {})
        s.update(ctx)
        samples.append(s)
    mined = _mine_common_traits(samples)
    llm_strats = _propose_strategies(samples)
    strats = mined + llm_strats
    written = 0
    persisted: List[Dict[str, Any]] = []
    for s in strats[:5]:
        definition = s.get("definition_json") or {}
        metrics = s.get("backtest_metrics") or {}
        backtest = metrics if isinstance(metrics, dict) else {}
        if "side" not in definition and s.get("side"):
            definition["side"] = s.get("side")
        if not backtest:
            try:
                base = definition.copy()
                base.setdefault("side", s.get("side"))
                bt = _backtest_strategy(base, contexts)
                s["backtest_metrics"] = bt
                backtest = bt
            except Exception as e:  # noqa: BLE001
                logger.debug(f"alpha backtest failed: {e}")
                backtest = {}
        if not backtest.get("signals", 0):
            continue
        try:
            name = str(s.get("name") or f"alpha_strategy_{written+1}")
            upsert_alpha_strategy(name, definition, backtest, "active")
            written += 1
            persisted.append({"name": name, "metrics": backtest, "definition": definition})
            try:
                push_values(
                    job="alpha_hunter",
                    values={
                        "strategy_expectancy": float(backtest.get("expectancy", 0.0) or 0.0),
                        "strategy_signals": float(backtest.get("signals", 0) or 0.0),
                    },
                    labels={"strategy": name},
                )
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            logger.debug(f"alpha strategy upsert failed: {e}")
            continue
    return {
        "status": "ok",
        "trades": len(trades),
        "snapshots": len(contexts),
        "strategies": written,
        "persisted": persisted,
        "mined_candidates": mined,
    }


@dataclass
class AlphaMatchResult:
    matched: bool
    score: float
    strategy_name: Optional[str] = None
    matched_rules: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


def _rules_match(rules: Iterable[Dict[str, Any]], feat_row: Dict[str, Any]) -> Tuple[bool, int]:
    ops = {
        '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
        '<': lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
    }
    matched = 0
    total = 0
    for rule in rules:
        total += 1
        k = str(rule.get("key"))
        op = str(rule.get("op"))
        if op not in ops or k not in feat_row:
            return False, matched
        try:
            val = float(feat_row[k])
            ok = ops[op](val, float(rule.get("value", 0.0)))
        except Exception:
            return False, matched
        if not ok:
            return False, matched
        matched += 1
    return total > 0 and matched == total, matched


def match_strategies_on_features(feat_row: Dict[str, Any]) -> AlphaMatchResult:
    try:
        strategies = fetch_active_alpha_strategies()
    except Exception:
        strategies = []
    best: Optional[AlphaMatchResult] = None
    for s in strategies:
        rules = (s.get("definition") or {}).get("selection_rules") or []
        ok, matched = _rules_match(rules, feat_row)
        if not ok:
            continue
        metrics = s.get("metrics") or {}
        score = float(metrics.get("expectancy", metrics.get("avg_return", 0.0)) or 0.0)
        result = AlphaMatchResult(
            matched=True,
            score=score,
            strategy_name=s.get("name"),
            matched_rules=rules,
            metadata={"signals": metrics.get("signals"), "win_rate": metrics.get("win_rate")},
        )
        if best is None or result.score > best.score:
            best = result
    if best:
        try:
            push_values(
                job="alpha_hunter",
                values={"match_score": float(best.score), "match": 1.0},
                labels={"strategy": str(best.strategy_name or "unknown")},
            )
        except Exception:
            pass
        return best
    return AlphaMatchResult(False, 0.0)


def _mine_common_traits(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not samples:
        return []
    num_samples = len(samples)
    min_support = max(2, int(max(num_samples * 0.6, 1)))
    feature_buckets: Dict[str, List[float]] = {}
    agent_flags: Counter[str] = Counter()
    sides = Counter()
    sources = Counter()
    for s in samples:
        side_val = str(s.get("side") or "").upper()
        if side_val:
            sides[side_val] += 1
        src = str(s.get("source") or "")
        if src:
            sources[src] += 1
        meta = s.get("meta") or {}
        for key in ["whale_signal", "smc_signal"]:
            if str(meta.get(key) or "").upper():
                agent_flags[key] += 1
        feat_vec = s.get("feature_vector") or {}
        for k, v in feat_vec.items():
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                feature_buckets.setdefault(k, []).append(fv)
    majority_side, _ = sides.most_common(1)[0] if sides else ("LONG", 1)
    selected_rules: List[Dict[str, Any]] = []
    notable_features: List[str] = []
    for feature, values in feature_buckets.items():
        if len(values) < min_support:
            continue
        spread = max(values) - min(values)
        if spread <= 1e-6:
            continue
        high_thr = _percentile(values, 65)
        low_thr = _percentile(values, 35)
        high_support = sum(1 for v in values if v >= high_thr)
        low_support = sum(1 for v in values if v <= low_thr)
        rule: Optional[Dict[str, Any]] = None
        if high_support >= min_support:
            rule = {"key": feature, "op": ">=", "value": round(high_thr, 6)}
        elif low_support >= min_support:
            rule = {"key": feature, "op": "<=", "value": round(low_thr, 6)}
        if rule:
            selected_rules.append(rule)
            notable_features.append(feature)
        if len(selected_rules) >= 6:
            break
    if not selected_rules:
        return []
    description = (
        "Derived from leaderboard trades: "
        + ", ".join(notable_features[:4])
    )
    strategy = {
        "name": f"alpha_leaderboard_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "description": description,
        "definition_json": {
            "selection_rules": selected_rules,
            "side": majority_side,
            "min_support": min_support,
            "sources": list(x for x, _ in sources.most_common(3)),
            "agents": [k for k, v in agent_flags.items() if v >= min_support],
            "horizon_minutes": int(os.getenv("ALPHA_DEFAULT_HORIZON_MIN", "45")),
        },
    }
    return [strategy]


def _evaluate_rules_on_df(df, rules: List[Dict[str, Any]], horizon_min: int, side: str) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"signals": 0}
    signals = 0
    returns: List[float] = []
    close = df["close"].astype(float)
    step = max(1, horizon_min)
    try:
        fee_bps = float(os.getenv("ALPHA_FEE_BPS", "2.0"))
    except Exception:
        fee_bps = 2.0
    try:
        slippage_bps = float(os.getenv("ALPHA_SLIPPAGE_BPS", "5.0"))
    except Exception:
        slippage_bps = 5.0
    cost = (fee_bps + slippage_bps) / 10000.0
    for idx in range(len(df) - step):
        row = df.iloc[idx]
        ok, _ = _rules_match(rules, row.to_dict())
        if not ok:
            continue
        entry = float(close.iloc[idx])
        future = float(close.iloc[idx + step])
        if future <= 0 or entry <= 0:
            continue
        if side == "SHORT":
            ret = (entry - future) / entry
        else:
            ret = (future - entry) / entry
        ret -= cost
        returns.append(ret)
        signals += 1
    if not returns:
        return {"signals": 0}
    wins = sum(1 for r in returns if r > 0)
    losses = len(returns) - wins
    avg_ret = sum(returns) / len(returns)
    avg_win = sum(r for r in returns if r > 0) / wins if wins else 0.0
    avg_loss = sum(r for r in returns if r <= 0) / losses if losses else 0.0
    expectancy = avg_win * (wins / len(returns)) + avg_loss * (losses / len(returns))
    stdev = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    sharpe = (avg_ret / stdev) * math.sqrt(len(returns)) if stdev > 1e-9 else 0.0
    return {
        "signals": int(signals),
        "win_rate": round(wins / len(returns), 3),
        "avg_return": round(avg_ret, 4),
        "expectancy": round(expectancy, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "sharpe": round(sharpe, 3),
    }


def _backtest_strategy(definition: Dict[str, Any], contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    rules = definition.get("selection_rules") or []
    if not rules:
        return {"signals": 0}
    horizon = int(definition.get("horizon_minutes") or os.getenv("ALPHA_DEFAULT_HORIZON_MIN", "45"))
    side = str(definition.get("side") or "LONG").upper()
    agg_returns: Dict[str, float] = {}
    totals = {"signals": 0, "wins": 0, "returns": []}  # type: ignore[var-annotated]
    win_rates: List[float] = []
    for ctx in contexts:
        df = ctx.get("df")
        metrics = _evaluate_rules_on_df(df, rules, horizon, side)
        if metrics.get("signals", 0):
            totals["signals"] += metrics["signals"]
            win_rates.append(metrics.get("win_rate", 0.0) or 0.0)
            agg_returns.setdefault("avg_return", 0.0)
            agg_returns["avg_return"] += metrics.get("avg_return", 0.0) * metrics["signals"]
            agg_returns.setdefault("avg_win", 0.0)
            agg_returns["avg_win"] += metrics.get("avg_win", 0.0) * metrics["signals"]
            agg_returns.setdefault("avg_loss", 0.0)
            agg_returns["avg_loss"] += metrics.get("avg_loss", 0.0) * metrics["signals"]
            totals.setdefault("returns", []).extend([metrics.get("avg_return", 0.0)] * metrics["signals"])
    if totals["signals"] == 0:
        return {"signals": 0}
    avg_return = agg_returns["avg_return"] / totals["signals"] if totals["signals"] else 0.0
    avg_win = agg_returns["avg_win"] / totals["signals"] if totals["signals"] else 0.0
    avg_loss = agg_returns["avg_loss"] / totals["signals"] if totals["signals"] else 0.0
    expectancy = avg_return
    sharpe = 0.0
    try:
        returns = totals.get("returns", [])
        if returns:
            stdev = statistics.pstdev(returns) if len(returns) > 1 else 0.0
            if stdev > 1e-9:
                sharpe = (statistics.mean(returns) / stdev) * math.sqrt(len(returns))
    except Exception:
        sharpe = 0.0
    return {
        "signals": int(totals["signals"]),
        "win_rate": round(sum(win_rates) / len(win_rates), 3) if win_rates else 0.0,
        "avg_return": round(avg_return, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "expectancy": round(expectancy, 4),
        "sharpe": round(sharpe, 3),
        "side": side,
        "horizon_minutes": horizon,
    }


def main() -> None:
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.agents.alpha_hunter '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = CollectInput(**json.loads(sys.argv[1]))
    out = run_collect_and_mine(payload)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
