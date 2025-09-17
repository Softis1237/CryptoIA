from __future__ import annotations

import io
import sys
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.reporting.charts import plot_price_with_smc_zones


def test_plot_price_with_smc_zones_smoke(monkeypatch, tmp_path):
    # minimal OHLCV frame
    df = pd.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5],
            "open": [100, 101, 102, 101, 103],
            "high": [101, 102, 103, 103, 104],
            "low": [99, 100, 101, 100, 102],
            "close": [100, 102, 101.5, 102.5, 103],
            "volume": [10, 11, 9, 12, 13],
        }
    )
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    # stub S3 IO
    monkeypatch.setattr(
        "pipeline.reporting.charts.download_bytes", lambda _: buf.getvalue()
    )
    out_files = {}

    def _fake_upload(path: str, content: bytes, content_type: str | None = None):  # noqa: D401
        out_files[path] = content
        return f"s3://test/{path}"

    monkeypatch.setattr("pipeline.reporting.charts.upload_bytes", _fake_upload)

    zones = [
        {"zone_type": "OB_BULL", "price_low": 100.5, "price_high": 101.5, "status": "untested"}
    ]
    uri = plot_price_with_smc_zones("s3://dummy/features.parquet", zones=zones, title="t", slot="test")
    assert uri.startswith("s3://test/")
    # ensure something was uploaded
    assert any(k.endswith("smc_zones.png") for k in out_files.keys())
