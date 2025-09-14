import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pipeline.features.features_patterns import detect_patterns


def test_detect_doji():
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.2, 2.2, 3.2],
            "low": [0.8, 1.8, 2.8],
            "close": [1.1, 2.1, 3.01],
        }
    )
    res = detect_patterns(df)
    assert res["pat_doji"].iloc[-1] == 1
    assert res["pat_score"].iloc[-1] == 1


def test_detect_harami_bull():
    df = pd.DataFrame(
        {
            "open": [1.2, 1.05],
            "high": [1.25, 1.15],
            "low": [0.9, 1.0],
            "close": [1.0, 1.1],
        }
    )
    res = detect_patterns(df)
    assert res["pat_harami_bull"].iloc[-1] == 1
    assert res["pat_score"].iloc[-1] == 1


def test_detect_three_white_soldiers():
    df = pd.DataFrame(
        {
            "open": [1.0, 1.05, 1.15],
            "high": [1.15, 1.25, 1.35],
            "low": [0.95, 1.0, 1.1],
            "close": [1.1, 1.2, 1.3],
        }
    )
    res = detect_patterns(df)
    assert res["pat_three_white_soldiers"].iloc[-1] == 1
    assert res["pat_score"].iloc[-1] == 1


def test_detect_head_and_shoulders():
    n = 60
    price = [100.0] * n
    price[9] = 99.0
    price[10] = 105.0
    price[11] = 99.0
    price[20] = 95.0
    price[24] = 99.0
    price[25] = 110.0
    price[26] = 99.0
    price[40] = 95.0
    price[49] = 99.0
    price[50] = 105.0
    price[51] = 99.0
    price[-1] = 94.0
    df = pd.DataFrame(
        {
            "open": price,
            "high": [p + 0.5 for p in price],
            "low": [p - 0.5 for p in price],
            "close": price,
        }
    )
    res = detect_patterns(df)
    assert res["pat_head_and_shoulders_recent"].iloc[-1] == 1
    assert res["pat_score"].iloc[-1] >= 1


def test_detect_triangle():
    n = 60
    high = pd.Series([100 - i for i in range(n)])
    low = pd.Series([60 + i for i in range(n)])
    close = (high + low) / 2
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    res = detect_patterns(df)
    assert res["pat_triangle_recent"].iloc[-1] == 1
    assert res["pat_score"].iloc[-1] == 1


def test_detect_flag():
    n = 60
    trend = list(pd.concat([
        pd.Series(np.linspace(100, 130, n // 2)),
        pd.Series(np.linspace(130, 135, n // 2)),
    ]))
    df = pd.DataFrame(
        {
            "open": trend,
            "high": [p + 1 for p in trend],
            "low": [p - 1 for p in trend],
            "close": trend,
        }
    )
    res = detect_patterns(df)
    assert res["pat_flag_recent"].iloc[-1] == 1
    assert res["pat_score"].iloc[-1] >= 1

