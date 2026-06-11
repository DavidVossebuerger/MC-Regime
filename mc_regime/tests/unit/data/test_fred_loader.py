"""Tests for the FRED data loader.

Tests that don't require FRED API are unit tests with synthetic series.
Tests that need FRED are marked with `requires_fred_key` and skip if
the env var is not set (so CI doesn't fail).
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from mc_regime.data.fred_loader import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FRED_SERIES,
    FREDRateLimitError,
    compute_yoy,
    load_fred_long,
    load_fred_panel,
    load_fred_yoy,
    load_parquet,
    save_parquet,
)


# --- Unit tests (no FRED needed) ---

def test_default_fred_series_has_5_currencies():
    currencies = {k[0] for k in DEFAULT_FRED_SERIES.keys()}
    assert currencies == {"USD", "EUR", "GBP", "JPY", "CHF"}


def test_default_fred_series_usd_has_core_macro():
    usd_indicators = {k[1] for k in DEFAULT_FRED_SERIES if k[0] == "USD"}
    assert "CPI" in usd_indicators
    assert "Fed Funds" in usd_indicators
    assert "Unemployment" in usd_indicators


def test_compute_yoy_constant_series_is_zero():
    s = pd.Series([100.0] * 24, index=pd.date_range("2020-01-01", periods=24, freq="ME"))
    yoy = compute_yoy(s)
    # First 12 are NaN, rest are 0
    assert yoy.iloc[12:].abs().max() < 1e-9


def test_compute_yoy_growing_series():
    # 12% compound growth over 12 months: each month multiplies by 1.12^(1/12)
    n = 24
    monthly_factor = 1.12 ** (1 / 12)
    vals = [100.0 * (monthly_factor ** i) for i in range(n)]
    s = pd.Series(vals, index=pd.date_range("2020-01-01", periods=n, freq="ME"))
    yoy = compute_yoy(s)
    # 12% YoY after 12 months
    assert abs(yoy.iloc[12] - 12.0) < 0.1


# --- FREDClient tests with mocking ---

def test_fredclient_rotates_on_rate_limit(monkeypatch):
    """When the primary key hits 429, the client should rotate to backup."""
    # Skip if no env vars
    if not os.environ.get("FRED_API_KEY"):
        pytest.skip("FRED_API_KEY not set")
    from fredapi import Fred
    # Mock Fred to simulate: primary raises on get_series, backup succeeds
    mock_primary = MagicMock()
    mock_backup = MagicMock()
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="ME"))
    mock_primary.get_series.side_effect = Exception("429 Too Many Requests")
    mock_backup.get_series.return_value = s
    with patch("fredapi.Fred") as MockFred:
        MockFred.side_effect = [mock_primary, mock_backup]
        from mc_regime.data.fred_loader import FREDClient
        client = FREDClient()
        result = client.get_series("TEST_ID")
        assert mock_primary.get_series.called
        assert mock_backup.get_series.called
        assert len(result) == 3


def test_fredclient_raises_when_both_keys_rate_limited(monkeypatch):
    if not os.environ.get("FRED_API_KEY"):
        pytest.skip("FRED_API_KEY not set")
    mock_primary = MagicMock()
    mock_backup = MagicMock()
    mock_primary.get_series.side_effect = Exception("429 Too Many Requests")
    mock_backup.get_series.side_effect = Exception("429 Too Many Requests")
    with patch("fredapi.Fred") as MockFred:
        MockFred.side_effect = [mock_primary, mock_backup]
        from mc_regime.data.fred_loader import FREDClient
        client = FREDClient()
        with pytest.raises(FREDRateLimitError):
            client.get_series("TEST_ID")


# --- Integration tests with real FRED (skipped without key) ---

FRED_INTEGRATION = pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"),
    reason="FRED_API_KEY not set; skipping live FRED tests"
)


@FRED_INTEGRATION
def test_fredclient_fetches_real_us_cpi():
    from mc_regime.data.fred_loader import FREDClient
    client = FREDClient()
    s = client.get_series("CPIAUCSL", start="2020-01-01")
    assert len(s) > 0
    assert s.iloc[-1] > 200  # US CPI > 200 in 2020+


@FRED_INTEGRATION
def test_fetch_indicator_real_data():
    from mc_regime.data.fred_loader import FREDClient, fetch_indicator
    client = FREDClient()
    s = fetch_indicator(client, "USD", "Fed Funds", start="2003-01-01")
    assert len(s) > 100  # ~20 years monthly
    assert s.mean() < 10  # Fed Funds between 0-10% in modern era


# --- Parquet roundtrip ---

def test_save_load_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="ME"),
        "value": np.random.rand(10),
    })
    path = tmp_path / "fred.parquet"
    save_parquet(df, path)
    loaded = load_parquet(path)
    pd.testing.assert_frame_equal(df, loaded)


# --- Cache loaders ---

def test_load_fred_panel_from_csv(tmp_path):
    """Loader reads a wide-format FRED CSV with date as first column."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "USD_CPI": [100.0, 100.5, 101.0, 101.2, 101.5],
        "EUR_CPI": [90.0, 90.1, 90.3, 90.5, 90.7],
    })
    path = tmp_path / "test_panel.csv"
    df.to_csv(path, index=False)
    loaded = load_fred_panel(path)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert "USD_CPI" in loaded.columns
    assert "EUR_CPI" in loaded.columns
    assert len(loaded) == 5


def test_load_fred_yoy_from_csv(tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "USD_CPI_YoY": [2.1, 2.2, 2.3, 2.4, 2.5],
    })
    path = tmp_path / "test_yoy.csv"
    df.to_csv(path, index=False)
    loaded = load_fred_yoy(path)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert "USD_CPI_YoY" in loaded.columns


def test_load_fred_long_from_csv(tmp_path):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
        "series": ["USD_CPI", "EUR_CPI", "USD_CPI"],
        "value": [100.0, 90.0, 100.5],
    })
    path = tmp_path / "test_long.csv"
    df.to_csv(path, index=False)
    loaded = load_fred_long(path)
    assert len(loaded) == 3
    assert {"date", "series", "value"}.issubset(set(loaded.columns))
    assert isinstance(loaded["date"].dtype, pd.DatetimeTZDtype) or "datetime" in str(loaded["date"].dtype)


@FRED_INTEGRATION
def test_real_cache_roundtrip(tmp_path):
    """Fetch a few series, write to CSV cache, reload from CSV."""
    from mc_regime.data.fred_loader import FREDClient
    client = FREDClient()
    s1 = client.get_series("CPIAUCSL", start="2020-01-01")
    s2 = client.get_series("FEDFUNDS", start="2020-01-01")
    # Write as cache-style wide CSV
    panel = pd.concat([s1.rename("USD_CPI"), s2.rename("USD_Fed Funds")], axis=1)
    panel.index.name = "date"
    cache_path = tmp_path / "fred_panel_levels.csv"
    panel.reset_index().to_csv(cache_path, index=False)
    loaded = load_fred_panel(cache_path)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert "USD_CPI" in loaded.columns
    assert "USD_Fed Funds" in loaded.columns
    assert len(loaded) == len(panel)
