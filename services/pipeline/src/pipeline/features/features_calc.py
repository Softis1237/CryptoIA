from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..infra.s3 import download_bytes, upload_bytes
from .features_cpd import enrich_with_cpd_simple
from .features_kats import enrich_with_kats


class FeaturesCalcInput(BaseModel):
    prices_path_s3: str
    news_signals: List[dict] = Field(default_factory=list)
    news_facts: List[dict] = Field(default_factory=list)
    # Optional enrichments
    orderbook_meta: Optional[Dict[str, Any]] = None
    orderflow_path_s3: Optional[str] = None
    onchain_signals: List[dict] = Field(default_factory=list)
    social_signals: List[dict] = Field(default_factory=list)
    regime_hint: Optional[str] = None  # if known from previous run
    macro_flags: List[str] = Field(default_factory=list)  # e.g., ["CPI", "FOMC"]
    social_window_minutes: int = 180
    run_id: Optional[str] = None
    slot: Optional[str] = None


class FeaturesCalcOutput(BaseModel):
    features_path_s3: str
    feature_schema: List[str]
    snapshot_ts: str


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = (
        pd.Series(up, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    )
    roll_down = (
        pd.Series(down, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    )
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat(
        [(high - low), (high - close).abs(), (low - close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / (ma.replace(0, np.nan)).abs()
    return upper, lower, width


def _stochastic(
    df: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(k_window).min()
    high_max = df["high"].rolling(k_window).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["close"] - low_min) / denom
    d = k.rolling(d_window).mean()
    return k, d


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_v = df["volume"].replace(0, np.nan).cumsum()
    cum_tp_v = (tp * df["volume"]).cumsum()
    return (cum_tp_v / cum_v).fillna(method="bfill")


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [float((df["open"].iloc[0] + df["close"].iloc[0]) / 2.0)]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha["ha_close"].iloc[i - 1]) / 2.0)
    ha["ha_open"] = pd.Series(ha_open, index=df.index)
    ha["ha_high"] = pd.concat([df["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(
        axis=1
    )
    ha["ha_low"] = pd.concat([df["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(
        axis=1
    )
    return ha


def _volume_profile_features(
    df: pd.DataFrame, window: int = 240, bins: int = 24
) -> Dict[str, float]:
    """Approximate Volume Profile on last window using bar VWAP as proxy per-bar price.

    Returns dict with vpoc price, deviation from close, and value area width.
    """
    try:
        x = df.tail(window).copy()
        if len(x) < max(20, bins):
            return {"vpoc_dev_pct": 0.0, "va_width_pct": 0.0}
        # proxy price per bar: typical price
        price = (
            x["high"].astype(float) + x["low"].astype(float) + x["close"].astype(float)
        ) / 3.0
        vol = x.get("volume", pd.Series([0.0] * len(x), index=x.index)).astype(float)
        pmin, pmax = float(price.min()), float(price.max())
        if not (pmax > pmin and vol.sum() > 0):
            return {"vpoc_dev_pct": 0.0, "va_width_pct": 0.0}
        hist, edges = np.histogram(
            price.values, bins=bins, range=(pmin, pmax), weights=vol.values
        )
        if hist.sum() <= 0:
            return {"vpoc_dev_pct": 0.0, "va_width_pct": 0.0}
        idx = int(np.argmax(hist))
        vpoc = float((edges[idx] + edges[idx + 1]) / 2.0)
        close = float(x["close"].iloc[-1])
        vpoc_dev_pct = (close - vpoc) / max(1e-9, close)
        # value area: 70% of volume around VPOC by greedy expansion
        target = 0.7 * hist.sum()
        left_idx = right_idx = idx
        total = hist[idx]
        while total < target and (left_idx > 0 or right_idx < len(hist) - 1):
            # expand to the side with larger next bin
            left_val = hist[left_idx - 1] if left_idx > 0 else -1
            right_val = hist[right_idx + 1] if right_idx < len(hist) - 1 else -1
            if right_val >= left_val:
                right_idx = min(len(hist) - 1, right_idx + 1)
                total += hist[right_idx]
            else:
                left_idx = max(0, left_idx - 1)
                total += hist[left_idx]
        va_low = float(edges[left_idx])
        va_high = float(edges[right_idx + 1])
        va_width_pct = (va_high - va_low) / max(1e-9, close)
        return {
            "vpoc_dev_pct": float(vpoc_dev_pct),
            "va_width_pct": float(va_width_pct),
        }
    except Exception:
        return {"vpoc_dev_pct": 0.0, "va_width_pct": 0.0}


def run(payload: FeaturesCalcInput) -> FeaturesCalcOutput:
    raw = download_bytes(payload.prices_path_s3)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas()
    # Assume single symbol for MVP
    df = df.sort_values("ts").reset_index(drop=True)

    # Technical features
    df["ema_20"] = _ema(df["close"], 20)
    df["ema_50"] = _ema(df["close"], 50)
    df["rsi_14"] = _rsi(df["close"], 14)
    df["atr_14"] = _atr(df, 14)
    df["ret_1m"] = df["close"].pct_change().fillna(0.0)

    # Advanced indicators
    macd, macd_sig, macd_hist = _macd(df["close"].astype(float))
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist

    bb_u, bb_l, bb_w = _bollinger(df["close"].astype(float), 20, 2.0)
    df["bb_upper"] = bb_u
    df["bb_lower"] = bb_l
    df["bb_width"] = bb_w.fillna(0.0)

    k, d = _stochastic(df)
    df["stoch_k"] = k.fillna(50.0)
    df["stoch_d"] = d.fillna(50.0)

    if "volume" in df.columns:
        df["vwap"] = _vwap(df)
        # volume z-score (rolling 50)
        vol = df["volume"].astype(float)
        df["volume_z"] = (vol - vol.rolling(50).mean()) / (
            vol.rolling(50).std(ddof=0) + 1e-9
        )
    else:
        df["vwap"] = df["close"]
        df["volume_z"] = 0.0

    ha = _heikin_ashi(df)
    for c in ["ha_open", "ha_close", "ha_high", "ha_low"]:
        df[c] = ha[c]

    # Volume profile features (window 240 bars)
    vp = _volume_profile_features(df, window=240, bins=24)
    df["vpoc_dev_pct"] = vp.get("vpoc_dev_pct", 0.0)
    df["va_width_pct"] = vp.get("va_width_pct", 0.0)

    # Calendar/cyc features
    dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    hour = dt.dt.hour.astype(int)
    dow = dt.dt.dayofweek.astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    # One-hot minimal (hour buckets and weekday) — keep compact: buckets of 6 hours
    hour_bucket = (hour // 6).astype(int)
    for b in range(4):
        df[f"hour_bucket_{b}"] = (hour_bucket == b).astype(int)
    for d in range(7):
        df[f"dow_{d}"] = (dow == d).astype(int)
    # Slot one-hot (limited known set)
    slot = (payload.slot or "manual").lower()
    df["slot_manual"] = 1 if slot == "manual" else 0
    df["slot_scheduled"] = 1 if slot == "scheduled" else 0
    df["slot_other"] = 1 if slot not in {"manual", "scheduled"} else 0

    # News features (MVP: aggregate last N signals)
    pos = sum(1 for s in payload.news_signals if s.get("sentiment") == "positive")
    neg = sum(1 for s in payload.news_signals if s.get("sentiment") == "negative")
    imp = sum(float(s.get("impact_score", 0.0)) for s in payload.news_signals)
    df["news_pos_count"] = pos
    df["news_neg_count"] = neg
    df["news_impact_sum"] = imp

    # News topics features (from LLM) - one-hot encode top N topics
    defined_topics = [
        "etf",
        "sec",
        "regulation",
        "hack",
        "exploit",
        "adoption",
        "fomc",
        "inflation",
        "grayscale",
        "blackrock",
        "binance",
        "coinbase",
        "stablecoin",
    ]
    try:
        topic_counts = {f"news_topic_{topic}_count": 0 for topic in defined_topics}

        all_topics = []
        for s in payload.news_signals:
            if s.get("topics") and isinstance(s.get("topics"), list):
                all_topics.extend([topic.lower() for topic in s.get("topics")])

        for topic in defined_topics:
            topic_counts[f"news_topic_{topic}_count"] = all_topics.count(topic)

        for key, value in topic_counts.items():
            df[key] = value

    except Exception:
        # If anything fails, just create zero-filled columns for schema consistency
        for topic in defined_topics:
            df[f"news_topic_{topic}_count"] = 0

    # News facts features (LLM) — weighted sums and counts per type/direction
    try:
        import math as _m

        # Map src_id -> impact_score for weighting
        _impact_map = {
            str(s.get("id")): float(s.get("impact_score", 0.0) or 0.0)
            for s in (payload.news_signals or [])
        }
        now = datetime.now(timezone.utc)
        decay_h = float(os.getenv("NEWS_FACTS_DECAY_HOURS", "12"))
        types = [
            "SEC_ACTION",
            "ETF_APPROVAL",
            "HACK",
            "FORK",
            "LISTING",
            "MACRO",
            "RUMOR_RETRACTION",
            "OTHER",
        ]
        dirs = ["bull", "bear", "volatility", "neutral"]
        sums: Dict[str, float] = {}
        cnts: Dict[str, int] = {}
        for f in payload.news_facts or []:
            try:
                t = str(f.get("type", "OTHER")).upper()
                if t not in types:
                    t = "OTHER"
                d = str(f.get("direction", "neutral")).lower()
                if d not in dirs:
                    d = "neutral"
                src = str(f.get("src_id", ""))
                imp_w = float(_impact_map.get(src, 1.0))
                mag = float(f.get("magnitude", 0.0) or 0.0)
                conf = float(f.get("confidence", 0.0) or 0.0)
                ts = str(f.get("ts", ""))
                try:
                    ts_dt = pd.Timestamp(ts).to_pydatetime().astimezone(timezone.utc)
                except Exception:
                    ts_dt = now
                age_h = max(0.0, (now - ts_dt).total_seconds() / 3600.0)
                decay = _m.exp(-age_h / max(0.1, decay_h)) if decay_h > 0 else 1.0
                w = imp_w * mag * conf * decay
                key_sum = f"fact_{t.lower()}_{d}_sum"
                key_cnt = f"fact_{t.lower()}_{d}_cnt"
                sums[key_sum] = sums.get(key_sum, 0.0) + w
                cnts[key_cnt] = cnts.get(key_cnt, 0) + 1
            except Exception:
                continue
        # Write into df as scalar features
        fact_cols: List[str] = []
        for k, v in sums.items():
            df[k] = float(v)
            fact_cols.append(k)
        for k, v in cnts.items():
            df[k] = float(v)
            fact_cols.append(k)
    except Exception:
        fact_cols = []  # type: ignore[assignment]

    # Orderbook features (if provided as meta)
    ob = payload.orderbook_meta or {}
    if ob:
        bid_vol = float(ob.get("bid_vol") or 0.0)
        ask_vol = float(ob.get("ask_vol") or 0.0)
        depth = float(ob.get("depth") or 0.0)
        imbalance = float(ob.get("imbalance") or 0.0)
        depth_ratio = bid_vol / (ask_vol + 1e-9) if (bid_vol or ask_vol) else 1.0
        df["ob_imbalance"] = imbalance
        df["ob_depth_ratio"] = depth_ratio
        df["ob_total_liq"] = bid_vol + ask_vol
        df["ob_depth"] = depth
        df["ob_bid_levels"] = float(ob.get("bid_n") or 0.0)
        df["ob_ask_levels"] = float(ob.get("ask_n") or 0.0)
        df["ob_orders_count"] = df["ob_bid_levels"] + df["ob_ask_levels"]
    else:
        for c in [
            "ob_imbalance",
            "ob_depth_ratio",
            "ob_total_liq",
            "ob_depth",
            "ob_bid_levels",
            "ob_ask_levels",
            "ob_orders_count",
        ]:
            df[c] = 0.0

    # On-chain aggregated features (from signals list)
    oc_map = {
        str(s.get("metric")): float(s.get("value", 0.0) or 0.0)
        for s in (payload.onchain_signals or [])
    }
    df["onchain_netflow_24h"] = oc_map.get("exchanges_netflow_sum", 0.0)
    df["onchain_active_addr"] = oc_map.get("active_addresses", 0.0)
    df["onchain_mvrv_z"] = oc_map.get("mvrv_z_score", 0.0)
    df["onchain_sopr"] = oc_map.get("sopr", 0.0)
    df["onchain_miners_balance"] = oc_map.get("miners_balance_sum", 0.0)
    df["onchain_transfers_vol"] = oc_map.get("transfers_volume_sum", 0.0)

    # Social metrics (tweets/posts per minute in last window)
    if payload.social_signals:
        try:
            now = datetime.now(timezone.utc)
            win = timedelta(minutes=max(10, int(payload.social_window_minutes)))

            def _parse_ts(s):
                try:
                    return pd.Timestamp(s).to_pydatetime().astimezone(timezone.utc)
                except Exception:
                    return now

            recent = [
                s
                for s in payload.social_signals
                if (now - _parse_ts(s.get("ts"))).total_seconds() <= win.total_seconds()
            ]
            tw = [s for s in recent if (s.get("platform") == "twitter")]
            rd = [s for s in recent if (s.get("platform") == "reddit")]
            minutes = max(1.0, win.total_seconds() / 60.0)
            df["tweets_per_min"] = len(tw) / minutes
            df["reddit_per_min"] = len(rd) / minutes
            # Burstiness proxy: 30m rate vs window rate
            win2 = timedelta(minutes=min(30, int(payload.social_window_minutes)))
            recent2 = [
                s
                for s in payload.social_signals
                if (now - _parse_ts(s.get("ts"))).total_seconds()
                <= win2.total_seconds()
            ]
            tw2 = [s for s in recent2 if (s.get("platform") == "twitter")]
            rd2 = [s for s in recent2 if (s.get("platform") == "reddit")]
            m2 = max(1.0, win2.total_seconds() / 60.0)
            tw_rate2 = len(tw2) / m2
            rd_rate2 = len(rd2) / m2
            df["tweets_burst"] = (
                tw_rate2 / max(1e-6, df["tweets_per_min"]) - 1.0
            ).replace([np.inf, -np.inf], 0.0)
            df["reddit_burst"] = (
                rd_rate2 / max(1e-6, df["reddit_per_min"]) - 1.0
            ).replace([np.inf, -np.inf], 0.0)
            # Average sentiment score if available
            try:
                s_vals = [float(s.get("score", 0.0) or 0.0) for s in recent]
                df["social_sent_mean"] = (
                    (sum(s_vals) / max(1, len(s_vals))) if s_vals else 0.0
                )
            except Exception:
                df["social_sent_mean"] = 0.0
        except Exception:
            df["tweets_per_min"] = 0.0
            df["reddit_per_min"] = 0.0
            df["tweets_burst"] = 0.0
            df["reddit_burst"] = 0.0
            df["social_sent_mean"] = 0.0
    else:
        df["tweets_per_min"] = 0.0
        df["reddit_per_min"] = 0.0
        df["tweets_burst"] = 0.0
        df["reddit_burst"] = 0.0
        df["social_sent_mean"] = 0.0

    # Regime hint one-hot (if provided)
    regimes = ["trend_up", "trend_down", "range", "volatility", "crash", "healthy_rise"]
    rh = (payload.regime_hint or "").lower()
    for r in regimes:
        df[f"regime_{r}"] = 1 if rh == r else 0

    # If no regime_hint, derive simple heuristic from EMAs/ATR to populate one-hot
    if not payload.regime_hint:
        try:
            close = df["close"].astype(float)
            ema20 = df.get("ema_20", close.ewm(span=20, adjust=False).mean()).astype(
                float
            )
            ema50 = df.get("ema_50", close.ewm(span=50, adjust=False).mean()).astype(
                float
            )
            atr = df.get("atr_14").astype(float)
            trend = float(
                ((ema20 - ema50) / (close.abs() + 1e-9))
                .ewm(span=20, adjust=False)
                .mean()
                .iloc[-1]
            )
            vol = float((atr.iloc[-1] / max(1e-9, close.iloc[-1])) * 100.0)
            label = "range"
            if vol > 3.5:
                label = "volatility"
            elif trend > 0.001:
                label = "trend_up"
            elif trend < -0.001:
                label = "trend_down"
            for r in regimes:
                df[f"regime_{r}"] = 1 if r == label else 0
        except Exception:
            pass

    # Macro flags presence count (if any)
    df["macro_flags_cnt"] = len(payload.macro_flags or [])

    # Integral news context score (0..1) using existing columns
    try:
        import math as _m

        a = float(os.getenv("NEWS_CTX_COEFF_A", "0.1"))
        b = float(os.getenv("NEWS_CTX_COEFF_B", "0.6"))
        c = float(os.getenv("NEWS_CTX_COEFF_C", "0.5"))
        burst = 0.0
        try:
            burst = float(
                max(
                    float(
                        df.get("tweets_burst", 0.0).iloc[-1]
                        if "tweets_burst" in df.columns
                        else 0.0
                    ),
                    float(
                        df.get("reddit_burst", 0.0).iloc[-1]
                        if "reddit_burst" in df.columns
                        else 0.0
                    ),
                )
            )
        except Exception:
            burst = 0.0
        # Use scalars for stability (replicated across rows)
        x = a * float(imp) + b * burst + c * float(len(payload.macro_flags or []))
        df["news_ctx_score"] = 1.0 / (1.0 + _m.exp(-float(x)))
    except Exception:
        df["news_ctx_score"] = 0.0

    # Order flow enrichment (optional): ofi/delta/throughput from last trades window
    try:
        if payload.orderflow_path_s3:
            from .features_order_flow import latest_metrics as _of_latest

            _of = _of_latest(payload.orderflow_path_s3)
            df["ofi_1m"] = float(_of.get("ofi_1m", 0.0) or 0.0)
            df["delta_vol_1m"] = float(_of.get("delta_vol_1m", 0.0) or 0.0)
            df["trades_per_sec"] = float(_of.get("trades_per_sec", 0.0) or 0.0)
            df["avg_trade_size"] = float(_of.get("avg_trade_size", 0.0) or 0.0)
        else:
            df["ofi_1m"] = 0.0
            df["delta_vol_1m"] = 0.0
            df["trades_per_sec"] = 0.0
            df["avg_trade_size"] = 0.0
    except Exception:
        df["ofi_1m"] = 0.0
        df["delta_vol_1m"] = 0.0
        df["trades_per_sec"] = 0.0
        df["avg_trade_size"] = 0.0

    # Optional Kats enrichment (cp/anomaly/seasonality)
    df, kats_meta = enrich_with_kats(df)
    # Optional CPD fallback (simple) if Kats off or additionally requested
    df, cpd_meta = enrich_with_cpd_simple(df)

    feature_cols = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_20",
        "ema_50",
        "rsi_14",
        "atr_14",
        "ret_1m",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "stoch_k",
        "stoch_d",
        "vwap",
        "volume_z",
        "ha_open",
        "ha_close",
        "ha_high",
        "ha_low",
        "vpoc_dev_pct",
        "va_width_pct",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "slot_manual",
        "slot_scheduled",
        "slot_other",
        "hour_bucket_0",
        "hour_bucket_1",
        "hour_bucket_2",
        "hour_bucket_3",
        "dow_0",
        "dow_1",
        "dow_2",
        "dow_3",
        "dow_4",
        "dow_5",
        "dow_6",
        "news_pos_count",
        "news_neg_count",
        "news_impact_sum",
        *(f"news_topic_{t}_count" for t in defined_topics),
        "news_ctx_score",
        "ob_imbalance",
        "ob_depth_ratio",
        "ob_total_liq",
        "ob_depth",
        "ob_bid_levels",
        "ob_ask_levels",
        "ob_orders_count",
        "onchain_netflow_24h",
        "onchain_active_addr",
        "onchain_mvrv_z",
        "onchain_sopr",
        "onchain_miners_balance",
        "onchain_transfers_vol",
        "tweets_per_min",
        "reddit_per_min",
        "tweets_burst",
        "reddit_burst",
        "social_sent_mean",
        "regime_trend_up",
        "regime_trend_down",
        "regime_range",
        "regime_volatility",
        "regime_crash",
        "regime_healthy_rise",
        "macro_flags_cnt",
        # order flow metrics (if present)
        "ofi_1m",
        "delta_vol_1m",
        "trades_per_sec",
        "avg_trade_size",
        # Kats flags (if present)
        *(["cp_flag"] if "cp_flag" in df.columns else []),
        *(["anomaly_flag"] if "anomaly_flag" in df.columns else []),
        *(["seasonality_count"] if "seasonality_count" in df.columns else []),
        *(["cp_simple_flag"] if "cp_simple_flag" in df.columns else []),
        *(["cpd_score"] if "cpd_score" in df.columns else []),
        # dynamically added facts features
        *(fact_cols if "fact_cols" in locals() else []),
    ]
    feat = df[feature_cols].copy()
    # Stabilize NaNs/Infs for downstream consumers
    try:
        num_cols = [c for c in feat.columns if c != "ts"]
        feat[num_cols] = feat[num_cols].replace([np.inf, -np.inf], np.nan)
        # Forward then backward fill to avoid gaps at window starts, fallback to 0.0
        feat[num_cols] = (
            feat[num_cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        )
    except Exception:
        pass

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slot = payload.slot or "manual"
    s3_path = f"runs/{date_key}/{slot}/features.parquet"

    import pyarrow as pa
    import pyarrow.parquet as pq

    table_out = pa.Table.from_pandas(feat)
    sink = pa.BufferOutputStream()
    pq.write_table(table_out, sink, compression="zstd")
    buf = sink.getvalue().to_pybytes()
    s3_uri = upload_bytes(s3_path, buf, content_type="application/octet-stream")

    out = FeaturesCalcOutput(
        features_path_s3=s3_uri,
        feature_schema=list(feat.columns),
        snapshot_ts=datetime.now(timezone.utc).isoformat(),
    )
    return out


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python -m pipeline.features.features_calc '<json_payload>'",
            file=sys.stderr,
        )
        sys.exit(2)
    payload_raw = sys.argv[1]
    payload = FeaturesCalcInput.model_validate_json(payload_raw)
    out = run(payload)
    print(out.model_dump_json())


if __name__ == "__main__":
    main()
