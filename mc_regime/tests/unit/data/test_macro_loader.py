"""Tests for the macro time-series loader."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from mc_regime.data.macro_loader import (
    DEFAULT_EVENTS,
    extract_all_macro,
    extract_macro_series,
    load_calendar,
    load_macro_wide,
    macro_wide_table,
    save_macro_wide,
)

CALENDAR_DIR = Path("/root/research/MC-Regime/Economic-Calendar")


def test_extract_macro_series_us_cpi():
    cal = load_calendar(CALENDAR_DIR)
    ts = extract_macro_series(cal, "USD", "CPI y/y")
    assert not ts.empty
    assert {"timestamp", "actual_str", "value"}.issubset(set(ts.columns))
    # USD CPI y/y is reported ~12 times per year
    n_years = (ts["timestamp"].max() - ts["timestamp"].min()).days / 365
    n_per_year = len(ts) / n_years
    assert 8 < n_per_year < 14


def test_extract_macro_series_returns_monthly_grid():
    cal = load_calendar(CALENDAR_DIR)
    ts = extract_macro_series(cal, "USD", "Unemployment Rate")
    # Should be sorted and unique
    assert ts["timestamp"].is_monotonic_increasing
    assert not ts["timestamp"].duplicated().any()


def test_extract_macro_series_filters_impact():
    cal = load_calendar(CALENDAR_DIR)
    # CPI y/y for USD is High-impact; min_impact=High should give same result
    high_only = extract_macro_series(cal, "USD", "CPI y/y", min_impact="High")
    medium_up = extract_macro_series(cal, "USD", "CPI y/y", min_impact="Medium")
    assert len(high_only) <= len(medium_up)


def test_extract_macro_series_empty_for_unknown():
    cal = load_calendar(CALENDAR_DIR)
    ts = extract_macro_series(cal, "USD", "DOES_NOT_EXIST")
    assert ts.empty


def test_extract_all_macro_returns_dict():
    out = extract_all_macro(CALENDAR_DIR)
    assert isinstance(out, dict)
    assert ("USD", "CPI y/y") in out
    assert ("EUR", "CPI Flash Estimate y/y") in out


def test_macro_wide_table_has_expected_columns():
    wide = macro_wide_table(CALENDAR_DIR)
    assert "timestamp" in wide.columns
    assert "USD_CPI y/y" in wide.columns
    assert "EUR_CPI Flash Estimate y/y" in wide.columns
    # Should have monthly resolution
    assert len(wide) > 50  # ~8 years × 12 months


def test_macro_wide_table_is_sorted_and_unique():
    wide = macro_wide_table(CALENDAR_DIR)
    assert wide["timestamp"].is_monotonic_increasing
    assert not wide["timestamp"].duplicated().any()


def test_macro_wide_table_allows_nan():
    wide = macro_wide_table(CALENDAR_DIR)
    # Different events have different coverage; NaN is expected
    n_cols = wide.shape[1] - 1  # exclude timestamp
    nan_frac = wide.iloc[:, 1:].isna().mean()
    # At least some columns have non-trivial NaN fraction
    assert (nan_frac > 0).any()


def test_default_events_has_5_currencies():
    assert set(DEFAULT_EVENTS.keys()) == {"USD", "EUR", "GBP", "JPY", "CHF"}
    for currency, events in DEFAULT_EVENTS.items():
        assert len(events) > 0


def test_save_load_macro_wide_roundtrip(tmp_path):
    wide = macro_wide_table(CALENDAR_DIR)
    path = tmp_path / "macro_wide.parquet"
    save_macro_wide(wide, path)
    loaded = load_macro_wide(path)
    assert loaded.shape == wide.shape
    assert list(loaded.columns) == list(wide.columns)
