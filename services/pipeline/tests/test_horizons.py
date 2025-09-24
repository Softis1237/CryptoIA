from pipeline.utils.horizons import (
    horizon_to_minutes,
    horizon_to_timedelta,
    horizon_to_hours,
    normalize_horizons,
    default_forecast_horizons,
    minutes_to_horizon,
)


def test_horizon_conversions():
    assert horizon_to_minutes("30m") == 30
    assert horizon_to_minutes("4h") == 240
    assert horizon_to_minutes("24h") == 1440
    assert horizon_to_timedelta("1h").total_seconds() == 3600
    assert horizon_to_hours("12h") == 12.0
    assert minutes_to_horizon(30) == "30m"
    assert minutes_to_horizon(240) == "4h"


def test_normalize_horizons():
    assert normalize_horizons(["4h", "30m", "4h"]) == ["4h", "30m"]
    assert default_forecast_horizons()[0] == "30m"
    assert "24h" in default_forecast_horizons()
