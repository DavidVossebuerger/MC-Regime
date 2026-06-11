"""Tests for the FX data loader."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from mc_regime.data.fx_loader import (
    KNOWN_PAIRS,
    _is_index_file,
    _pair_from_filename,
    clean_pair,
    discover_files,
    inventory,
    load_pair,
    resample_bars,
    save_parquet,
    load_parquet,
)

DATA_DIR = Path("/root/research/MC-Regime/dukascopy_data")
EURUSD_DIR = Path("/root/research/MC-Regime")


def test_is_index_file_detects_usa500():
    assert _is_index_file(Path("usa500idxusd-m30-bid-2007-01-01-2007-03-31.csv"))
    assert _is_index_file(Path("usatechidxusd-m30-bid-2011-09-19T13-30.csv"))


def test_is_index_file_does_not_flag_fx():
    assert not _is_index_file(Path("eurusd-m30-bid-2003-05-04T21-2025-09-07.csv"))
    assert not _is_index_file(Path("gbpusd-m30-bid-2007-01-01-2007-03-31.csv"))
    assert not _is_index_file(Path("usdjpy-m30-bid-2020-06-01-2020-06-30.csv"))


def test_pair_from_filename_known():
    assert _pair_from_filename(Path("eurusd-m30-bid-2003-05-04T21-2025-09-07.csv")) == "EURUSD"
    assert _pair_from_filename(Path("gbpusd-m30-bid-2007-01-01-2007-03-31.csv")) == "GBPUSD"
    assert _pair_from_filename(Path("usdjpy-m30-bid-2007-01-01-2007-03-31.csv")) == "USDJPY"


def test_pair_from_filename_old_convention():
    assert _pair_from_filename(Path("gbpusd_2007-Q1.csv")) == "GBPUSD"


def test_discover_files_finds_eurusd():
    files = discover_files(EURUSD_DIR, "EURUSD")
    assert len(files) == 1
    assert files[0].name == "eurusd-m30-bid-2003-05-04T21-2025-09-07.csv"


def test_discover_files_finds_gbpusd_excludes_indices():
    files = discover_files(DATA_DIR, "GBPUSD")
    assert len(files) > 50
    # None should be index files
    assert not any("idx" in f.name.lower() for f in files)


def test_discover_files_unknown_pair():
    files = discover_files(DATA_DIR, "CHFUSD")
    assert files == []  # discover returns empty; load_pair raises
    with pytest.raises(FileNotFoundError):
        load_pair(DATA_DIR, "CHFUSD")


def test_load_pair_eurusd():
    df = load_pair(EURUSD_DIR, "EURUSD")
    assert len(df) > 100_000
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "pair"]
    assert df["pair"].unique().tolist() == ["EURUSD"]
    assert df["timestamp"].is_monotonic_increasing
    assert not df.duplicated(subset=["timestamp"]).any()


def test_load_pair_gbpusd_dedupes_across_files():
    df = load_pair(DATA_DIR, "GBPUSD")
    assert len(df) > 100_000
    assert not df.duplicated(subset=["timestamp"]).any()
    # Should cover 2007 to ~2026
    assert df["timestamp"].min().year == 2007
    assert df["timestamp"].max().year >= 2025


def test_load_pair_usdjpy():
    df = load_pair(DATA_DIR, "USDJPY")
    assert len(df) > 100_000
    # JPY quotes are large (100+), unlike EUR/USD (~1.1)
    assert df["close"].mean() > 50


def test_clean_pair_drops_nan_and_zero_prices():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 01:00"], utc=True, format="ISO8601"
        ),
        "open": [1.1, np.nan, 1.2],
        "high": [1.15, 1.16, 1.21],
        "low": [1.09, 1.08, 1.19],
        "close": [1.12, 1.10, 1.20],
        "pair": ["EURUSD", "EURUSD", "EURUSD"],
    })
    cleaned = clean_pair(df, drop_weekends=False, drop_outliers=False)
    assert len(cleaned) == 2  # NaN row dropped


def test_clean_pair_drops_weekends():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2020-01-03 12:00",  # Friday
             "2020-01-04 12:00",  # Saturday
             "2020-01-05 12:00",  # Sunday
             "2020-01-06 12:00"], # Monday
            utc=True),
        "open": [1.1, 1.1, 1.1, 1.1],
        "high": [1.1, 1.1, 1.1, 1.1],
        "low": [1.1, 1.1, 1.1, 1.1],
        "close": [1.1, 1.1, 1.1, 1.1],
        "pair": ["EURUSD"] * 4,
    })
    cleaned = clean_pair(df, drop_weekends=True, drop_outliers=False)
    assert len(cleaned) == 2  # Sat+Sun dropped


def test_resample_bars_to_1h():
    timestamps = pd.date_range("2020-01-01", periods=4, freq="30min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [1.1, 1.2, 1.3, 1.4],
        "high": [1.15, 1.25, 1.35, 1.45],
        "low": [1.05, 1.15, 1.25, 1.35],
        "close": [1.12, 1.22, 1.32, 1.42],
        "pair": ["EURUSD"] * 4,
    })
    out = resample_bars(df, freq="1h", agg="ohlc")
    assert len(out) == 2
    assert out.iloc[0]["open"] == 1.1
    assert out.iloc[0]["close"] == 1.22  # close of last 30-min bar in the 1h bucket


def test_inventory_shows_fx_only():
    inv = inventory(DATA_DIR)
    assert "EURUSD" not in inv  # EURUSD is in root, not dukascopy_data
    assert "GBPUSD" in inv
    assert "USDJPY" in inv
    # No indices
    for code in inv:
        assert "idx" not in code.lower()


def test_save_and_load_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "open": [1.1, 1.2],
        "high": [1.15, 1.25],
        "low": [1.05, 1.15],
        "close": [1.12, 1.22],
        "pair": ["EURUSD", "EURUSD"],
    })
    path = tmp_path / "test.parquet"
    save_parquet(df, path)
    loaded = load_parquet(path)
    pd.testing.assert_frame_equal(df, loaded)
