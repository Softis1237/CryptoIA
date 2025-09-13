from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

from loguru import logger

from ..data.ingest_order_flow import IngestOrderFlowInput, run as run_of
from .queue import publish_trigger, get_key, setex, incrby
from ..infra.db import get_conn


SYMBOL = os.getenv("TRIGGER_SYMBOL", os.getenv("PAPER_SYMBOL", "BTC/USDT"))
WINDOW_SEC = int(os.getenv("ORDERFLOW_WINDOW_SEC", "60"))
POLL_SLEEP = float(os.getenv("TRIGGER_POLL_SLEEP", "5.0"))
VOL_FACTOR = float(os.getenv("TRIGGER_VOL_SPIKE_FACTOR", "3.0"))
DELTA_ABS_MIN = float(os.getenv("TRIGGER_DELTA_BASE_MIN", "20.0"))  # in base units (e.g., BTC)
COOLDOWN_SEC = int(os.getenv("TRIGGER_COOLDOWN_SEC", "300"))
WATCH_NEWS = os.getenv("TRIGGER_WATCH_NEWS", "1") in {"1", "true", "True"}
NEWS_IMPACT_MIN = float(os.getenv("TRIGGER_NEWS_IMPACT_MIN", "0.7"))
ADAPTIVE_LEARNING = os.getenv("TRIGGER_ADAPTIVE", "1") in {"1", "true", "True"}


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


def _maybe_news_triggers() -> None:
    if not WATCH_NEWS:
        return
    try:
        # poll recent high-impact news signals
        since_sec = int(os.getenv("TRIGGER_NEWS_LOOKBACK_SEC", "120"))
        now = datetime.now(timezone.utc)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT src_id, ts, title, source, impact_score FROM news_signals WHERE ts >= %s AND impact_score >= %s ORDER BY ts DESC LIMIT 5",
                    (now - timedelta(seconds=since_sec), float(NEWS_IMPACT_MIN)),
                )
                rows = cur.fetchall() or []
        for src_id, ts, title, source, score in rows:
            key = f"rt:news:seen:{src_id}"
            if get_key(key):
                continue
            setex(key, 86400, "1")
            if _cooldown_ok("NEWS"):
                publish_trigger(
                    {
                        "type": "NEWS",
                        "ts": int(getattr(ts, "timestamp", lambda: _now_ts())()),
                        "symbol": SYMBOL,
                        "meta": {
                            "impact_score": float(score or 0.0),
                            "title": str(title or ""),
                            "source": str(source or ""),
                            "src_id": str(src_id or ""),
                        },
                    }
                )
    except Exception:
        # best-effort
        pass


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
    logger.info("TriggerAgent started (always-on)")
    delta_buf: List[float] = []
    last_adapt = 0
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
            # News triggers best-effort
            _maybe_news_triggers()
            # Periodically recompute priorities from paper PnL
            if (time.time() - last_adapt) > float(os.getenv("TRIGGER_ADAPT_EVERY_SEC", "300")):
                _update_trigger_priorities()
                last_adapt = time.time()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"TriggerAgent error: {e}")
        time.sleep(POLL_SLEEP)


if __name__ == "__main__":
    main()
