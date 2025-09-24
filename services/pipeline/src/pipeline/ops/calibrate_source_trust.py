from __future__ import annotations

"""Утилита для пересчёта коэффициентов доверия к источникам."""

import argparse
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from loguru import logger

try:  # pragma: no cover - зависимости используются по необходимости
    import pandas as pd
except Exception as exc:  # noqa: BLE001
    raise SystemExit("pandas is required: pip install pandas") from exc

try:
    import numpy as np
except Exception as exc:  # noqa: BLE001
    raise SystemExit("numpy is required: pip install numpy") from exc

try:
    import requests
except Exception as exc:  # noqa: BLE001
    raise SystemExit("requests is required: pip install requests") from exc


@dataclass(slots=True)
class CalibrationRow:
    source: str
    signal_correlation: float
    popularity: float


def _coingecko(days: int = 30) -> CalibrationRow:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    prices = pd.DataFrame(data.get("prices"), columns=["ts", "price"]).astype(float)
    volumes = pd.DataFrame(data.get("total_volumes"), columns=["ts", "volume"]).astype(float)
    df = prices.join(volumes.set_index("ts"), on="ts")
    df["return"] = df["price"].pct_change().fillna(0.0)
    df["volume_delta"] = df["volume"].pct_change().fillna(0.0)
    corr = df[["return", "volume_delta"]].corr().iloc[0, 1]
    community = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin", timeout=10).json()
    popularity = float(community.get("community_data", {}).get("reddit_subscribers", 0) or 0) / 5_000_000
    popularity = float(np.clip(popularity, 0.5, 1.0))
    return CalibrationRow("CoinGecko API", float(corr or 0.0), popularity)


def _cryptodatadownload(url: str) -> CalibrationRow:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    lines = [line for line in r.text.splitlines() if not line.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df = df.sort_values("date")
    df["return"] = df["close"].pct_change().fillna(0.0)
    df["volume_delta"] = df["Volume USDT"].pct_change().fillna(0.0)
    corr = df[["return", "volume_delta"]].corr().iloc[0, 1]
    popularity = min(0.8, 0.5 + (len(df) / 10_000))
    return CalibrationRow("CryptoDataDownload BTC", float(corr or 0.0), popularity)


def _default(name: str) -> CalibrationRow:
    return CalibrationRow(name, 0.4, 0.55)


def calibrate(catalog: Path) -> List[CalibrationRow]:
    rows: List[CalibrationRow] = []
    if not catalog.exists():
        raise FileNotFoundError(catalog)
    data = pd.read_json(catalog)
    for item in data.to_dict(orient="records"):
        name = item.get("name", "")
        provider = str(item.get("provider", "")).lower()
        try:
            if provider == "coingecko":
                rows.append(_coingecko())
            elif provider == "cryptodatadownload":
                rows.append(_cryptodatadownload(item["url"]))
            else:
                rows.append(_default(name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Calibration failed for %s: %s", name, exc)
            rows.append(_default(name))
    return rows


def write_csv(rows: Iterable[CalibrationRow], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "signal_correlation", "popularity"])
        for row in rows:
            writer.writerow([row.source, f"{row.signal_correlation:.4f}", f"{row.popularity:.4f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate trust scores for data sources")
    parser.add_argument("--catalog", default="data/sources_catalog.json")
    parser.add_argument("--output", default="data/source_trust_calibration.csv")
    args = parser.parse_args()
    catalog = Path(args.catalog)
    rows = calibrate(catalog)
    write_csv(rows, Path(args.output))
    logger.info("calibration saved: %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
