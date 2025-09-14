import unittest
import os
import sys
from pathlib import Path

import importlib.util
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ensure src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline.features.features_supply_demand import latest_supply_demand_metrics  # noqa: E402
from pipeline.features.features_calc import FeaturesCalcInput, run  # noqa: E402


class TestSupplyDemandFeatures(unittest.TestCase):
    def test_latest_supply_demand_metrics(self):
        meta = {
            "orderbook": {"bid_vol": 120, "ask_vol": 80},
            "derivatives": {"open_interest": 5000, "funding_rate": 0.01},
            "onchain": {"long_term_holders": 700, "short_term_holders": 300},
        }
        m = latest_supply_demand_metrics(meta)
        self.assertAlmostEqual(m["orderbook_imbalance"], 0.2, places=6)
        self.assertEqual(m["open_interest"], 5000)
        self.assertEqual(m["funding_rate"], 0.01)
        self.assertAlmostEqual(m["long_term_holder_ratio"], 0.7, places=6)
        self.assertAlmostEqual(m["short_term_holder_ratio"], 0.3, places=6)

    def test_features_calc_integration(self):
        # prepare price dataframe
        df = pd.DataFrame(
            {
                "ts": list(range(10)),
                "open": [1 + i for i in range(10)],
                "high": [1.5 + i for i in range(10)],
                "low": [0.5 + i for i in range(10)],
                "close": [1 + i for i in range(10)],
                "volume": [100] * 10,
            }
        )
        table = pa.Table.from_pandas(df)
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        price_bytes = sink.getvalue().to_pybytes()

        # patch download/upload
        import pipeline.features.features_calc as fc

        fc.download_bytes = lambda path: price_bytes  # type: ignore
        captured = {}

        def fake_upload(path, data, content_type=""):
            captured["bytes"] = data
            return "s3://test/features.parquet"

        fc.upload_bytes = fake_upload  # type: ignore
        fc.enrich_with_cpd_simple = lambda df: (df, {})  # type: ignore
        fc.enrich_with_kats = lambda df: (df, {})  # type: ignore
        fc.detect_patterns = lambda df, patterns: {}  # type: ignore
        fc.load_patterns_from_db = lambda: []  # type: ignore
        fc._volume_profile_features = lambda df, window=240, bins=24: {"vpoc_dev_pct": 0.0, "va_width_pct": 0.0}  # type: ignore

        meta = {
            "orderbook": {"bid_vol": 120, "ask_vol": 80},
            "derivatives": {"open_interest": 5000, "funding_rate": 0.01},
            "onchain": {"long_term_holders": 700, "short_term_holders": 300},
        }
        payload = FeaturesCalcInput(prices_path_s3="s3://dummy", supply_demand_meta=meta)
        out = run(payload)
        self.assertIn("sd_open_interest", out.feature_schema)
        # read captured dataframe
        buf = captured["bytes"]
        table_out = pq.read_table(pa.BufferReader(buf))
        df_out = table_out.to_pandas()
        self.assertAlmostEqual(df_out["sd_orderbook_imbalance"].iloc[0], 0.2, places=6)
        self.assertEqual(df_out["sd_open_interest"].iloc[0], 5000)
        self.assertEqual(df_out["sd_funding_rate"].iloc[0], 0.01)
        self.assertAlmostEqual(df_out["sd_long_term_ratio"].iloc[0], 0.7, places=6)
        self.assertAlmostEqual(df_out["sd_short_term_ratio"].iloc[0], 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
