from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

from loguru import logger

from ..data.ingest_order_flow import IngestOrderFlowInput, run as run_of
from .queue import publish_trigger, get_key, setex, incrby
from ..infra.db import get_conn
from ..infra.metrics import push_values
from ..infra.health import start_background as start_health_server
from ..data.ingest_news import IngestNewsInput as _NI, run as _run_news
from ..infra.db import insert_news_signals as _insert_news, insert_news_facts as _insert_facts
import ccxt


SYMBOL = os.getenv("TRIGGER_SYMBOL", os.getenv("PAPER_SYMBOL", "BTC/USDT"))
WINDOW_SEC = int(os.getenv("ORDERFLOW_WINDOW_SEC", "60"))
POLL_SLEEP = float(os.getenv("TRIGGER_POLL_SLEEP", "5.0"))
VOL_FACTOR = float(os.getenv("TRIGGER_VOL_SPIKE_FACTOR", "3.0"))
DELTA_ABS_MIN = float(os.getenv("TRIGGER_DELTA_BASE_MIN", "20.0"))  # in base units (e.g., BTC)
COOLDOWN_SEC = int(os.getenv("TRIGGER_COOLDOWN_SEC", "300"))
WATCH_NEWS = os.getenv("TRIGGER_WATCH_NEWS", "1") in {"1", "true", "True"}
NEWS_IMPACT_MIN = float(os.getenv("TRIGGER_NEWS_IMPACT_MIN", "0.7"))
ADAPTIVE_LEARNING = os.getenv("TRIGGER_ADAPTIVE", "1") in {"1", "true", "True"}

# L2 orderbook settings
L2_ENABLE = os.getenv("TRIGGER_L2_ENABLE", "1") in {"1", "true", "True"}
L2_DEPTH_LEVELS = int(os.getenv("TRIGGER_L2_DEPTH_LEVELS", "50"))
L2_NEAR_BPS = float(os.getenv("TRIGGER_L2_NEAR_BPS", "10"))
L2_WALL_MIN_BASE = float(os.getenv("TRIGGER_L2_WALL_MIN_BASE", "50.0"))
L2_IMBALANCE_RATIO = float(os.getenv("TRIGGER_L2_IMBALANCE_RATIO", "2.0"))
L2_PROVIDER = os.getenv("FUTURES_PROVIDER", os.getenv("CCXT_PROVIDER", "binance"))
L2_USE_WS = os.getenv("ORDERFLOW_USE_WS", "0") in {"1", "true", "True"}  # reuse flag

# Lightweight news polling
NEWS_POLL_SEC = float(os.getenv("TRIGGER_NEWS_POLL_SEC", "120"))
NEWS_POLL_WINDOW_H = int(os.getenv("TRIGGER_NEWS_WINDOW_H", "1"))


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _ema_update(key: str, x: float, alpha: float = 0.05) -> float:
    try:
        cur = get_key(key)
        val = float(cur) if cur is not None else x
    except Exception:
        val = x
    new = (1.0 - alpha) * val + alpha * float(x)
    setex(key, 24 * 3600, str(new))
    return new


def _maybe_trigger_volume(meta: Dict) -> Tuple[bool, Dict]:
    total_vol = float(meta.get("total_vol", 0.0) or 0.0)
    ema = _ema_update("rt:vol_ema", total_vol, alpha=float(os.getenv("TRIGGER_VOL_EMA_ALPHA", "0.03")))
    # Adaptive adjustment
    factor = VOL_FACTOR
    if ADAPTIVE_LEARNING:
        try:
            adj = get_key("rt:prio:VOL_SPIKE")
            if adj is not None:
                factor = max(1.0, VOL_FACTOR / max(0.1, float(adj)))
        except Exception:
            pass
    ratio = (total_vol / ema) if ema > 0 else 0.0
    return (ratio >= factor), {"ratio": ratio, "total_vol": total_vol, "ema": ema}


def _maybe_trigger_delta(meta: Dict, window_buf: List[float]) -> Tuple[bool, Dict]:
    # meta has buy_vol/sell_vol in base units (e.g., BTC)
    buy_v = float(meta.get("buy_vol", 0.0) or 0.0)
    sell_v = float(meta.get("sell_vol", 0.0) or 0.0)
    delta = (buy_v - sell_v)
    window_buf.append(delta)
    # keep ~5 minutes aggregate
    keep = max(1, int(5 * 60 // max(1, int(WINDOW_SEC))))
    while len(window_buf) > keep:
        window_buf.pop(0)
    agg = sum(window_buf)
    # Adaptive adjustment: reduce threshold if strong positive priority
    thr = DELTA_ABS_MIN
    if ADAPTIVE_LEARNING:
        try:
            adj = get_key("rt:prio:DELTA_SPIKE")
            if adj is not None and float(adj) > 1.0:
                thr = max(5.0, DELTA_ABS_MIN / float(adj))
        except Exception:
            pass
    return (abs(agg) >= thr), {"delta_agg": agg, "thr": thr, "last_window": delta}


def _cooldown_ok(kind: str) -> bool:
    last = get_key(f"rt:last:{kind}")
    now = _now_ts()
    try:
        if last is not None and (now - int(last)) < COOLDOWN_SEC:
            return False
    except Exception:
        pass
    setex(f"rt:last:{kind}", COOLDOWN_SEC, str(now))
    return True


def _publish(kind: str, meta: Dict) -> None:
    if not _cooldown_ok(kind):
        return
    ev = {"type": kind, "ts": _now_ts(), "symbol": SYMBOL, "meta": meta}
    ok = publish_trigger(ev)
    if ok:
        logger.info(f"RT trigger → {kind}: {meta}")
        # Metrics: emit event
        try:
            push_values(job="rt_trigger", values={f"fired_{kind}": 1.0}, labels={"symbol": SYMBOL})
        except Exception:
            pass


def _maybe_news_triggers() -> None:
    if not WATCH_NEWS:
        return
    try:
        # poll recent high-impact news signals from DB first
        since_sec = int(os.getenv("TRIGGER_NEWS_LOOKBACK_SEC", "120"))
        now = datetime.now(timezone.utc)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT src_id, ts, title, source, impact_score FROM news_signals WHERE ts >= %s AND impact_score >= %s ORDER BY ts DESC LIMIT 10",
                    (now - timedelta(seconds=since_sec), float(NEWS_IMPACT_MIN)),
                )
                rows = cur.fetchall() or []
        for src_id, ts, title, source, score in rows:
            key = f"rt:news:seen:{src_id}"
            if get_key(key):
                continue
            setex(key, 86400, "1")
            _publish(
                "NEWS",
                {
                    "impact_score": float(score or 0.0),
                    "title": str(title or ""),
                    "source": str(source or ""),
                    "src_id": str(src_id or ""),
                },
            )
    except Exception:
        # best-effort
        pass


def _maybe_poll_news_lightweight(last_polled: list[float]) -> None:
    if not WATCH_NEWS:
        return
    import time as _t
    now = _t.time()
    if last_polled and (now - last_polled[0]) < NEWS_POLL_SEC:
        return
    last_polled[:] = [now]
    try:
        # Use the existing ingestion but it’s lightweight if LLM flags are off
        rid = str(_now_ts())
        out = _run_news(_NI(run_id=rid, slot="rt", time_window_hours=max(1, NEWS_POLL_WINDOW_H)))
        sigs = getattr(out, "news_signals", []) or []
        rows = []
        for s in sigs:
            url = getattr(s, "url", "")
            src_id = url or f"{s.ts}-{(s.title or '')[:28]}"
            rows.append(
                {
                    "src_id": src_id,
                    "ts": datetime.fromtimestamp(int(s.ts), tz=timezone.utc),
                    "title": s.title,
                    "source": s.source,
                    "url": url,
                    "sentiment": s.sentiment,
                    "topics": s.topics or [],
                    "impact_score": float(s.impact_score or 0.0),
                    "confidence": None,
                }
            )
        if rows:
            _insert_news(rows)
        facts = (getattr(out, "news_facts", None) or [])
        if facts:
            # normalize for insert function
            ins = []
            for f in facts:
                ins.append(
                    {
                        "src_id": str(f.get("src_id") or ""),
                        "ts": datetime.fromtimestamp(int(f.get("ts", _now_ts())), tz=timezone.utc),
                        "type": str(f.get("type") or "OTHER"),
                        "direction": str(f.get("direction") or "neutral"),
                        "magnitude": float(f.get("magnitude", 0.0) or 0.0),
                        "confidence": float(f.get("confidence", 0.0) or 0.0),
                        "entities": f.get("entities") or {},
                        "raw": f,
                    }
                )
            if ins:
                _insert_facts(ins)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"news poll failed: {e}")


def _check_l2_triggers() -> None:
    if not L2_ENABLE:
        return
    try:
        market = SYMBOL
        ex = getattr(ccxt, L2_PROVIDER)({"enableRateLimit": True})
        ob = ex.fetch_order_book(market, limit=max(5, L2_DEPTH_LEVELS))
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        near = (L2_NEAR_BPS / 10000.0) * mid
        # Walls: largest level within near distance
        wall_buy = max((amt for price, amt in bids if (mid - price) <= near), default=0.0)
        wall_sell = max((amt for price, amt in asks if (price - mid) <= near), default=0.0)
        if wall_buy >= L2_WALL_MIN_BASE or wall_sell >= L2_WALL_MIN_BASE:
            side = "BUY" if wall_buy >= wall_sell else "SELL"
            size = float(max(wall_buy, wall_sell))
            _publish("L2_WALL", {"side": side, "size_base": size, "near_bps": L2_NEAR_BPS})
        # Imbalance: sum of top-K levels
        k = max(5, min(L2_DEPTH_LEVELS, 50))
        sum_b = float(sum(amt for _, amt in bids[:k]))
        sum_a = float(sum(amt for _, amt in asks[:k]))
        ratio = (sum_b / max(1e-9, sum_a)) if sum_a > 0 else (float("inf") if sum_b > 0 else 0.0)
        if ratio >= L2_IMBALANCE_RATIO or (sum_a / max(1e-9, sum_b)) >= L2_IMBALANCE_RATIO:
            _publish("L2_IMBALANCE", {"bid_sum": sum_b, "ask_sum": sum_a, "ratio": ratio, "top_levels": k})
    except Exception as e:  # noqa: BLE001
        logger.debug(f"L2 check failed: {e}")


def _update_trigger_priorities() -> None:
    if not ADAPTIVE_LEARNING:
        return
    try:
        from datetime import timedelta
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT t.reason_codes, pnl.realized_pnl
                    FROM paper_pnl pnl
                    JOIN paper_positions p USING (pos_id)
                    JOIN trades_suggestions t ON (t.run_id = p.meta_json->>'run_id')
                    WHERE p.opened_at >= now() - interval '7 days'
                    """
                )
                rows = cur.fetchall() or []
        acc: Dict[str, float] = {}
        for rc_list, pnl in rows:
            try:
                codes = rc_list or []
                for c in codes:
                    s = str(c)
                    if s.startswith("TRIGGER:"):
                        acc[s.replace("TRIGGER:", "")] = acc.get(s.replace("TRIGGER:", ""), 0.0) + float(pnl or 0.0)
            except Exception:
                continue
        # Convert to positive priorities [0.5..2.0] around 1.0 baseline
        if acc:
            mx = max(abs(v) for v in acc.values()) or 1.0
            for k, v in acc.items():
                pr = 1.0 + 0.8 * (float(v) / mx)
                setex(f"rt:prio:{k}", 3600, str(max(0.5, min(2.0, pr))))
    except Exception:
        pass


def main() -> None:
    start_health_server()
    logger.info("TriggerAgent started (always-on)")
    delta_buf: List[float] = []
    last_adapt = 0
    _news_polled = [0.0]
    while True:
        try:
            # ingest order flow for WINDOW_SEC and compute summary
            out = run_of(
                IngestOrderFlowInput(
                    run_id=str(_now_ts()), slot="rt", symbol=SYMBOL.replace("/", ""), provider=os.getenv("FUTURES_PROVIDER", os.getenv("CCXT_PROVIDER", "binance")), window_sec=WINDOW_SEC
                )
            )
            meta = out.meta
            # Volume spike
            fired, info = _maybe_trigger_volume(meta)
            if fired:
                _publish("VOL_SPIKE", info)
            # Delta spike (agg 5m)
            fired2, info2 = _maybe_trigger_delta(meta, delta_buf)
            if fired2:
                _publish("DELTA_SPIKE", info2)
            # L2 triggers
            _check_l2_triggers()
            # News triggers best-effort
            _maybe_news_triggers()
            # Lightweight news polling (internal, writes to DB and may emit NEWS after that)
            _maybe_poll_news_lightweight(_news_polled)
            # Periodically recompute priorities from paper PnL
            if (time.time() - last_adapt) > float(os.getenv("TRIGGER_ADAPT_EVERY_SEC", "300")):
                _update_trigger_priorities()
                last_adapt = time.time()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"TriggerAgent error: {e}")
        time.sleep(POLL_SLEEP)


if __name__ == "__main__":
    main()
