# flake8: noqa
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cli(args):
    cmd = [sys.executable, "-m", "pipeline.cli.analysis", *args]
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(root)
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return json.loads(res.stdout.strip())


def test_price(tmp_path: Path):
    df = pd.DataFrame(
        {
            "ts": [0, 60, 120],
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10, 11, 12],
            "volume": [100, 110, 120],
        }
    )
    path = tmp_path / "prices.parquet"
    df.to_parquet(path)
    out = run_cli(["price", str(path)])
    assert "close" in out and "ema20" in out


def test_orderflow(tmp_path: Path):
    df = pd.DataFrame(
        {
            "ts": [0, 30, 60],
            "price": [10, 10.5, 11],
            "amount": [1, 2, 1.5],
            "side": ["buy", "sell", "buy"],
        }
    )
    path = tmp_path / "trades.parquet"
    df.to_parquet(path)
    out = run_cli(["orderflow", str(path)])
    assert "ofi_1m" in out


def test_supply_demand():
    meta = {
        "bid_vol": 100,
        "ask_vol": 80,
        "depth": 20,
        "imbalance": 0.1,
        "bid_n": 10,
        "ask_n": 8,
    }
    out = run_cli(["supply-demand", json.dumps(meta)])
    assert out["ob_total_liq"] == 180


def test_patterns(tmp_path: Path):
    df = pd.DataFrame(
        {
            "ts": [0, 60, 120],
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10, 12, 11],
            "volume": [100, 100, 100],
        }
    )
    path = tmp_path / "prices.parquet"
    df.to_parquet(path)
    out = run_cli(["patterns", str(path)])
    assert "pat_hammer" in out
