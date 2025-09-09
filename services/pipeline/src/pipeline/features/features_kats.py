from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd
from loguru import logger


def enrich_with_kats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Optionally enrich dataframe with Kats features: changepoints, anomalies, seasonality.

    Returns updated df and metadata dict. Safe to call without Kats installed.
    """
    if os.getenv("USE_KATS", "0") not in {"1", "true", "True"}:
        return df, {"kats_enabled": False}
    try:
        from kats.consts import TimeSeriesData  # type: ignore
        from kats.detectors.cusum_detection import CUSUMDetector  # type: ignore
        from kats.detectors.seasonality import SeasonalityDetector  # type: ignore
        try:
            from kats.detectors.outlier import OutlierDetector  # type: ignore
        except Exception:  # legacy/optional
            OutlierDetector = None  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Kats not available: {e}")
        return df, {"kats_enabled": False, "error": str(e)}

    try:
        work = df.copy()
        if "dt" not in work.columns:
            work["dt"] = pd.to_datetime(work["ts"], unit="s", utc=True)
        ts = TimeSeriesData(time=work["dt"].to_pydatetime(), value=work["close"].astype(float))

        # Seasonality
        try:
            sd = SeasonalityDetector(ts)
            seas = sd.detect_seasonality().get("seasonality", [])  # type: ignore
        except Exception:
            seas = []

        # Changepoints (CUSUM)
        try:
            cp = CUSUMDetector(ts, threshold=5.0)
            res_cp = cp.detector()
            cp_times = set([r["start_time"] for r in (res_cp.get("change_points", []) or [])])  # type: ignore
        except Exception:
            cp_times = set()

        # Outliers
        out_times = set()
        if OutlierDetector is not None:
            try:
                od = OutlierDetector(ts, outlier_type="additive")
                res_out = od.detector()
                out_times = set([r.time for r in (res_out or [])])  # type: ignore
            except Exception:
                out_times = set()

        # Map to dataframe flags
        work["cp_flag"] = work["dt"].isin(cp_times).astype(int)
        work["anomaly_flag"] = work["dt"].isin(out_times).astype(int)
        work["seasonality_count"] = int(len(seas))

        meta = {"kats_enabled": True, "seasonality_list": list(seas), "cp_count": int(len(cp_times)), "outlier_count": int(len(out_times))}
        return work, meta
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Kats enrichment failed: {e}")
        return df, {"kats_enabled": False, "error": str(e)}
