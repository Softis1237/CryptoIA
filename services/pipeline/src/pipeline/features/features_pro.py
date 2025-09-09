from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / (mid.replace(0, np.nan).abs())
    return upper, lower, width


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    eps = 1e-9
    k = 100.0 * (df["close"] - low_min) / (high_max - low_min + eps)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def heikin_ashi(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close


def vwap(df: pd.DataFrame, by_day: bool = True) -> pd.Series:
    price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    if by_day:
        # group by UTC date boundary
        day = pd.to_datetime(df["ts"], unit="s", utc=True).dt.floor("D")
        pv = (price * vol).groupby(day).cumsum()
        vv = vol.groupby(day).cumsum().replace(0.0, np.nan)
        return pv / vv
    # global cumulative VWAP as fallback
    pv = (price * vol).cumsum()
    vv = vol.cumsum().replace(0.0, np.nan)
    return pv / vv


def add_time_onehot(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    # Day of week one-hot (0..6)
    dow = dt.dt.weekday
    for d in range(7):
        df[f"dow_{d}"] = (dow == d).astype(int)
    # Hour of day one-hot (0..23)
    hour = dt.dt.hour
    for h in range(24):
        df[f"hour_{h}"] = (hour == h).astype(int)
    return df


def enrich_with_pro(df: pd.DataFrame) -> pd.DataFrame:
    if os.getenv("FEATURES_PRO", "1") not in {"1", "true", "True"}:
        return df
    work = df.copy()
    # MACD
    m, s, h = macd(work["close"].astype(float))
    work["macd"] = m
    work["macd_signal"] = s
    work["macd_hist"] = h
    # Bollinger
    bb_u, bb_l, bb_w = bollinger(work["close"].astype(float), window=20, n_std=2.0)
    work["bb_upper"] = bb_u
    work["bb_lower"] = bb_l
    work["bb_width"] = bb_w
    # Stochastic Oscillator
    k, d = stochastic_oscillator(work, k_period=14, d_period=3)
    work["stoch_k"] = k
    work["stoch_d"] = d
    # VWAP (per day)
    work["vwap"] = vwap(work, by_day=True)
    # Heikin-Ashi
    ha_o, ha_h, ha_l, ha_c = heikin_ashi(work)
    work["ha_open"] = ha_o
    work["ha_high"] = ha_h
    work["ha_low"] = ha_l
    work["ha_close"] = ha_c
    # Time one-hot
    work = add_time_onehot(work)
    # Clean NaNs at the start of windows
    num_cols = [c for c in work.columns if c not in {"ts", "ymbol"}]
    work[num_cols] = work[num_cols].fillna(method="ffill").fillna(method="bfill")