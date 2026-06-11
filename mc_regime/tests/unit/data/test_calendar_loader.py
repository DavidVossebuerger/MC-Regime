"""Tests for the economic calendar loader."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from mc_regime.data.calendar_loader import (
    _parse_numeric,
    compute_surprise_index,
    filter_by_currency,
    filter_by_impact,
    load_calendar,
    load_parquet,
    save_parquet,
)

CALENDAR_DIR = Path("/root/research/MC-Regime/Economic-Calendar")


# ---- Parser tests ----

def test_parse_numeric_plain():
    assert _parse_numeric("47.4") == 47.4
    assert _parse_numeric("-0.4") == -0.4
    assert _parse_numeric("164") == 164.0
    assert _parse_numeric("0.0") == 0.0


def test_parse_numeric_suffixes():
    assert _parse_numeric("164K") == 164_000
    assert _parse_numeric("8.79M") == 8_790_000
    assert _parse_numeric("1.2B") == 1.2e9
    assert _parse_numeric("0.5T") == 5e11


def test_parse_numeric_percent():
    assert _parse_numeric("0.1%") == 0.1
    assert _parse_numeric("-0.4%") == -0.4
    assert _parse_numeric("3.7%") == 3.7


def test_parse_numeric_range():
    val = _parse_numeric("1.63|3.0")
    assert abs(val - 2.315) < 1e-6


def test_parse_numeric_invalid():
    assert _parse_numeric(None) is None
    assert _parse_numeric(np.nan) is None
    assert _parse_numeric("") is None
    assert _parse_numeric("-") is None
    assert _parse_numeric("abc") is None


# ---- Loader tests ----

def test_load_calendar_returns_dataframe():
    cal = load_calendar(CALENDAR_DIR)
    assert isinstance(cal, pd.DataFrame)
    assert len(cal) > 10_000
    expected_cols = {"timestamp", "currency", "event", "impact",
                     "actual", "forecast", "previous",
                     "actual_num", "forecast_num", "previous_num", "surprise"}
    assert expected_cols.issubset(set(cal.columns))


def test_load_calendar_has_5_major_currencies():
    cal = load_calendar(CALENDAR_DIR)
    for c in ["USD", "EUR", "GBP", "JPY", "CHF"]:
        assert (cal["currency"] == c).sum() > 100, f"Missing {c}"


def test_load_calendar_timestamp_is_utc():
    cal = load_calendar(CALENDAR_DIR)
    assert cal["timestamp"].dt.tz is not None
    assert str(cal["timestamp"].dt.tz) == "UTC"


def test_load_calendar_dedupes():
    cal = load_calendar(CALENDAR_DIR)
    assert not cal.duplicated(subset=["timestamp", "currency", "event", "impact"]).any()


def test_load_calendar_surprise_is_actual_minus_forecast():
    cal = load_calendar(CALENDAR_DIR)
    rows = cal[cal["surprise"].notna()].head(20)
    for _, row in rows.iterrows():
        expected = row["actual_num"] - row["forecast_num"]
        assert abs(row["surprise"] - expected) < 1e-6


# ---- Filter tests ----

def test_filter_by_currency():
    cal = load_calendar(CALENDAR_DIR)
    sub = filter_by_currency(cal, ["USD", "EUR"])
    assert set(sub["currency"].unique()) == {"USD", "EUR"}


def test_filter_by_impact_medium_keeps_medium_and_high():
    cal = load_calendar(CALENDAR_DIR)
    sub = filter_by_impact(cal, "Medium")
    allowed = {"Medium", "High"}
    assert set(sub["impact"].unique()).issubset(allowed)


def test_filter_by_impact_high_only_keeps_high():
    cal = load_calendar(CALENDAR_DIR)
    sub = filter_by_impact(cal, "High")
    assert set(sub["impact"].unique()) == {"High"}


# ---- Surprise index tests ----

def test_compute_surprise_index_returns_dataframe():
    cal = load_calendar(CALENDAR_DIR)
    idx = compute_surprise_index(cal, "USD", freq="1D")
    assert not idx.empty
    assert {"surprise_raw", "surprise_index", "n_events"}.issubset(set(idx.columns))
    assert idx["timestamp"].is_monotonic_increasing


def test_compute_surprise_index_handles_empty_currency():
    cal = load_calendar(CALENDAR_DIR)
    idx = compute_surprise_index(cal, "XYZ_FAKE", freq="1D")
    assert idx.empty


def test_compute_surprise_index_normalized():
    cal = load_calendar(CALENDAR_DIR)
    idx = compute_surprise_index(cal, "USD", freq="1D", standardize=True)
    # After standardisation, long-run mean should be ~0 and std ~1
    valid = idx["surprise_index"].dropna()
    assert abs(valid.mean()) < 0.5
    assert 0.5 < valid.std() < 2.0


# ---- Parquet roundtrip ----

def test_save_load_parquet_roundtrip(tmp_path):
    cal = load_calendar(CALENDAR_DIR)
    path = tmp_path / "cal.parquet"
    save_parquet(cal.head(100), path)
    loaded = load_parquet(path)
    assert len(loaded) == 100
