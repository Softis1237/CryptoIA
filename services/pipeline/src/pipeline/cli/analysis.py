# flake8: noqa
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import pandas as pd

from ..features.features_calc import _atr, _ema, _rsi
from ..features.features_patterns import detect_patterns


def _read_bytes(path: str) -> bytes:
    if path.startswith("s3://"):
        from ..infra.s3 import download_bytes

        return download_bytes(path)
    with open(path, "rb") as f:
        return f.read()


def cmd_price(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.path).sort_values("ts").reset_index(drop=True)
    ema20 = _ema(df["close"], 20).iloc[-1]
    ema50 = _ema(df["close"], 50).iloc[-1]
    rsi14 = _rsi(df["close"], 14).iloc[-1]
    atr14 = _atr(df, 14).iloc[-1]
    out = {
        "close": float(df["close"].iloc[-1]),
        "ema20": float(ema20),
        "ema50": float(ema50),
        "rsi14": float(rsi14),
        "atr14": float(atr14),
    }
    print(json.dumps(out, ensure_ascii=False))


def cmd_orderflow(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.path).sort_values("ts").reset_index(drop=True)
    if df.empty:
        metrics = {
            "ofi_1m": 0.0,
            "delta_vol_1m": 0.0,
            "trades_per_sec": 0.0,
            "avg_trade_size": 0.0,
        }
    else:
        dt = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.set_index(dt)
        buy = df[df["side"] == "buy"]["amount"].resample("1min").sum().fillna(0.0)
        sell = df[df["side"] == "sell"]["amount"].resample("1min").sum().fillna(0.0)
        total = df["amount"].resample("1min").sum().fillna(0.0)
        count = df["amount"].resample("1min").count().fillna(0)
        ofi = (buy - sell) / total.replace(0.0, pd.NA)
        ofi = ofi.fillna(0.0)
        delta_vol = (buy - sell).fillna(0.0)
        trades_per_sec = (count / 60.0).astype(float)
        avg_trade_size = (total / count.replace(0, pd.NA)).fillna(0.0)
        last = (
            pd.DataFrame(
                {
                    "ofi": ofi,
                    "delta_vol": delta_vol,
                    "trades_per_sec": trades_per_sec,
                    "avg_trade_size": avg_trade_size,
                }
            )
            .reset_index()
            .tail(1)
            .iloc[0]
        )
        metrics = {
            "ofi_1m": float(last.get("ofi", 0.0) or 0.0),
            "delta_vol_1m": float(last.get("delta_vol", 0.0) or 0.0),
            "trades_per_sec": float(last.get("trades_per_sec", 0.0) or 0.0),
            "avg_trade_size": float(last.get("avg_trade_size", 0.0) or 0.0),
        }
    print(json.dumps(metrics, ensure_ascii=False))


def cmd_supply_demand(args: argparse.Namespace) -> None:
    meta: Dict[str, Any] = json.loads(args.json)
    bid_vol = float(meta.get("bid_vol", 0.0) or 0.0)
    ask_vol = float(meta.get("ask_vol", 0.0) or 0.0)
    depth = float(meta.get("depth", 0.0) or 0.0)
    imbalance = float(meta.get("imbalance", 0.0) or 0.0)
    bid_n = float(meta.get("bid_n", 0.0) or 0.0)
    ask_n = float(meta.get("ask_n", 0.0) or 0.0)
    depth_ratio = bid_vol / (ask_vol + 1e-9) if (bid_vol or ask_vol) else 1.0
    total_liq = bid_vol + ask_vol
    orders_count = bid_n + ask_n
    out = {
        "ob_imbalance": imbalance,
        "ob_depth_ratio": depth_ratio,
        "ob_total_liq": total_liq,
        "ob_depth": depth,
        "ob_bid_levels": bid_n,
        "ob_ask_levels": ask_n,
        "ob_orders_count": orders_count,
    }
    print(json.dumps(out, ensure_ascii=False))


def cmd_patterns(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.path).sort_values("ts").reset_index(drop=True)
    pat_map = detect_patterns(df)
    out = {k: float(v.iloc[-1]) for k, v in pat_map.items()}
    print(json.dumps(out, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Market analysis utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_price = sub.add_parser("price", help="Latest price metrics from parquet")
    p_price.add_argument("path", help="Path or S3 URI to prices parquet")
    p_price.set_defaults(func=cmd_price)

    p_of = sub.add_parser("orderflow", help="Order flow metrics from trades parquet")
    p_of.add_argument("path", help="Path or S3 URI to trades parquet")
    p_of.set_defaults(func=cmd_orderflow)

    p_sd = sub.add_parser("supply-demand", help="Metrics from orderbook meta JSON")
    p_sd.add_argument("json", help="Orderbook meta as JSON string")
    p_sd.set_defaults(func=cmd_supply_demand)

    p_pat = sub.add_parser("patterns", help="Detect candlestick patterns from parquet")
    p_pat.add_argument("path", help="Path or S3 URI to prices parquet")
    p_pat.set_defaults(func=cmd_patterns)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
