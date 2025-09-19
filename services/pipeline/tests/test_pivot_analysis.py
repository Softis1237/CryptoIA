import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.features.pivot_analysis import find_pivots
from pipeline.features.chart_patterns import (
    detect_head_and_shoulders,
    detect_head_and_shoulders_from_pivots,
    detect_inverse_head_and_shoulders_from_pivots,
)
from pipeline.features.features_patterns import detect_patterns


def test_find_pivots_basic_extrema():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 2, 1, 2, 3],
            "high": [1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5, 1.5, 0.5, 1.5, 2.5],
            "close": [1, 2, 3, 2, 1, 2, 3],
        }
    )

    piv = find_pivots(df, left_bars=1, right_bars=1)
    assert not piv.empty
    kinds = piv["kind"].tolist()
    assert "high" in kinds and "low" in kinds


def _make_classic_hs_dataframe() -> pd.DataFrame:
    close = [
        100.0,
        103.0,
        108.0,  # left shoulder high
        102.0,
        98.0,  # neckline low 1
        112.0,  # head high
        101.0,
        99.0,  # neckline low 2
        107.0,  # right shoulder high
        102.0,
        95.0,  # breakout close below neckline
        94.0,
    ]
    df = pd.DataFrame(
        {
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
        }
    )
    return df


def _make_inverse_hs_dataframe() -> pd.DataFrame:
    close = [
        120.0,
        119.0,
        113.0,  # left shoulder low
        116.0,
        118.0,  # neckline high 1
        114.0,
        109.0,  # head low
        115.0,
        119.0,  # neckline high 2
        114.0,
        112.0,  # right shoulder low
        116.0,
        115.0,  # last close still below neckline
    ]
    df = pd.DataFrame(
        {
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
        }
    )
    return df


def test_detect_head_and_shoulders_from_pivots_confirmed():
    df = _make_classic_hs_dataframe()
    piv = find_pivots(df, left_bars=2, right_bars=2)
    res = detect_head_and_shoulders_from_pivots(
        piv,
        df["close"],
        shoulder_tol=0.05,
        head_margin=0.01,
        min_separation=1,
        neckline_tol=0.0,
    )

    assert res.name == "head_and_shoulders"
    assert res.status == "confirmed"
    assert res.direction == "bearish"
    assert res.neckline is not None and res.breakout_price is not None
    assert res.left_shoulder is not None and res.right_shoulder is not None and res.head is not None


def test_detect_inverse_head_and_shoulders_in_progress():
    df = _make_inverse_hs_dataframe()
    piv = find_pivots(df, left_bars=1, right_bars=1)
    res = detect_inverse_head_and_shoulders_from_pivots(
        piv,
        df["close"],
        shoulder_tol=0.05,
        head_margin=0.01,
        min_separation=1,
        neckline_tol=0.0,
    )

    assert res.name == "inverse_head_and_shoulders"
    assert res.status == "in_progress"
    assert res.direction == "bullish"
    assert res.neckline is not None
    assert res.breakout_price is None
    assert res.meta["variant"] == "inverse"


def test_detect_head_and_shoulders_requires_input():
    with pytest.raises(ValueError):
        detect_head_and_shoulders(None)


def test_detect_patterns_talib_columns_present():
    df = pd.DataFrame(
        {
            "open": [1.0, 1.05, 1.1],
            "high": [1.1, 1.15, 1.2],
            "low": [0.95, 1.0, 1.05],
            "close": [1.02, 1.04, 1.06],
        }
    )

    res = detect_patterns(df)
    expected = {
        "pat_doji_ta",
        "pat_hanging_man_ta",
        "pat_shooting_star_ta",
        "pat_hammer_ta",
        "pat_inverted_hammer_ta",
        "pat_morning_star_ta",
        "pat_evening_star_ta",
    }

    for key in expected:
        assert key in res
        series = res[key]
        assert isinstance(series, pd.Series)
        assert len(series) == len(df)
