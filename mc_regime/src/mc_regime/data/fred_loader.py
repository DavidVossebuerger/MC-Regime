"""FRED data loader for macroeconomic time series.

Fetches official macroeconomic series from the Federal Reserve Economic
Data (FRED) database via the fredapi library. Handles two API keys with
automatic rotation on rate-limit (HTTP 429).

For repeated access (and to avoid rate limits), the data is cached as CSV
in `mc_regime/outputs/data/fred/`. The default workflow is:
    1. One-time fetch via `fetch_and_cache(force_refresh=True)`.
    2. Subsequent reads via `load_fred_panel(csv_path)` from the cache.

API keys are loaded from environment variables:
    FRED_API_KEY         (primary)
    FRED_API_KEY_BACKUP  (fallback)
These should be set in a local .env file (NOT committed).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dotenv import load_dotenv

# Load .env from project root if not already loaded
_PROJECT_ROOT = Path(__file__).parents[4]
load_dotenv(_PROJECT_ROOT / ".env")


# Default cache directory
DEFAULT_CACHE_DIR = _PROJECT_ROOT / "mc_regime" / "outputs" / "data" / "fred"


# Standard FRED series mapping for our 5 currencies.
# Each entry: (currency, indicator) -> FRED series id
# Series marked with a comment "NOT IN FRED" have been verified absent.
DEFAULT_FRED_SERIES: dict[tuple[str, str], str] = {
    # United States
    ("USD", "CPI"): "CPIAUCSL",                    # Headline CPI (level, monthly)
    ("USD", "Core CPI"): "CPILFESL",                # Core CPI (level, monthly)
    ("USD", "PCE"): "PCEPI",                       # PCE price index (level, monthly)
    ("USD", "Core PCE"): "PCEPILFE",                # Core PCE (level, monthly)
    ("USD", "Fed Funds"): "FEDFUNDS",              # Effective Fed Funds Rate (monthly)
    ("USD", "Unemployment"): "UNRATE",             # Unemployment rate (%)
    ("USD", "NFP"): "PAYEMS",                      # All Employees, Total Nonfarm (thousands)
    ("USD", "GDP"): "GDPC1",                       # Real GDP (billions of chained 2017 $)
    ("USD", "10Y Treasury"): "DGS10",              # 10-Year Treasury Constant Maturity (%)
    ("USD", "Industrial Production"): "INDPRO",   # Industrial Production Index

    # Euro Area (OECD mirrored in FRED)
    ("EUR", "CPI"): "CP0000EZ19M086NEST",          # HICP All Items (index)
    ("EUR", "Core CPI"): "00XTOBEZ19M086NEST",     # HICP ex-energy, food
    ("EUR", "Policy Rate"): "ECBDFR",              # ECB Deposit Facility Rate (daily)
    ("EUR", "Unemployment"): "LRHUTTTTEZM156S",    # Harmonized unemployment rate
    ("EUR", "GDP"): "CLVMNACSCAB1GQEA19",          # Real GDP (index)

    # United Kingdom
    ("GBP", "CPI"): "GBRCPIALLMINMEI",             # CPI All Items (index)
    ("GBP", "Core CPI"): "GBRCPICORMINMEI",        # Core CPI
    ("GBP", "Policy Rate"): "BOERUKM",             # Bank of England Policy Rate
    ("GBP", "Unemployment"): "LRHUTTTTGBM156S",    # Harmonized unemployment rate
    ("GBP", "GDP"): "NGDPRSAXDCGBQ",              # Real GDP for Great Britain (quarterly)

    # Japan
    ("JPY", "CPI"): "CPALTT01JPM657N",             # CPI All Items (index)
    # JPY Core CPI: not directly in FRED OECD mirror
    ("JPY", "Policy Rate"): "IRSTCB01JPM156N",     # BoJ Policy Rate (short-term)
    # JPY Unemployment: not directly in FRED OECD mirror
    ("JPY", "GDP"): "JPNRGDPEXP",                  # Real GDP for Japan (quarterly)

    # Switzerland
    ("CHF", "CPI"): "CHECPIALLMINMEI",             # CPI All Items (index)
    ("CHF", "Policy Rate"): "IRSTCB01CHM156N",     # SNB Policy Rate
    # CHF Unemployment: not directly in FRED OECD mirror
    ("CHF", "GDP"): "CLVMNACSAB1GQCH",             # Real GDP for Switzerland (quarterly)
}


class FREDRateLimitError(Exception):
    """Both primary and backup FRED keys hit rate limits."""


class FREDClient:
    """Thin wrapper around fredapi with two-key rotation on rate limit."""

    def __init__(self):
        from fredapi import Fred
        primary = os.environ.get("FRED_API_KEY")
        backup = os.environ.get("FRED_API_KEY_BACKUP")
        if not primary:
            raise RuntimeError(
                "FRED_API_KEY not set. Create a .env file in the project root "
                "with FRED_API_KEY=... and FRED_API_KEY_BACKUP=..."
            )
        self._primary = Fred(api_key=primary)
        self._backup = Fred(api_key=backup) if backup else None
        self._using_backup = False

    @property
    def _active(self):
        return self._backup if self._using_backup else self._primary

    def _maybe_rotate(self, exc: Exception) -> bool:
        """On rate limit, try the other key. Returns True if rotated, False if no fallback."""
        msg = str(exc).lower()
        is_rate = "rate" in msg or "429" in msg or "too many" in msg
        if not is_rate:
            return False
        if self._using_backup or self._backup is None:
            return False
        self._using_backup = True
        # Brief backoff to let primary cool down
        time.sleep(2)
        return True

    def get_series(
        self, series_id: str, start: str | None = None, end: str | None = None
    ) -> pd.Series:
        """Fetch a single FRED series, with automatic key rotation on 429."""
        kwargs = {}
        if start:
            kwargs["observation_start"] = start
        if end:
            kwargs["observation_end"] = end
        last_exc: Exception | None = None
        for attempt in range(2):  # primary, then backup
            try:
                return self._active.get_series(series_id, **kwargs)
            except Exception as e:
                last_exc = e
                if not self._maybe_rotate(e):
                    break
        raise FREDRateLimitError(
            f"Both FRED keys rate-limited fetching {series_id}: {last_exc}"
        )

    def search(self, text: str, limit: int = 10) -> pd.DataFrame:
        """Search FRED for series matching text, with key rotation."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return self._active.search(text, limit=limit)
            except Exception as e:
                last_exc = e
                if not self._maybe_rotate(e):
                    break
        raise FREDRateLimitError(f"Both FRED keys rate-limited on search: {last_exc}")


def fetch_macro_panel(
    series_map: dict[tuple[str, str], str] | None = None,
    start: str = "2003-01-01",
    end: str | None = None,
    max_retries: int = 2,
    retry_wait: float = 5.0,
    silent: bool = False,
) -> pd.DataFrame:
    """Fetch a panel of macroeconomic series from FRED and return as wide DataFrame.

    Args:
        series_map: {(currency, indicator): fred_id}. Defaults to DEFAULT_FRED_SERIES.
        start: ISO date for observation start.
        end: ISO date for observation end (None = latest).
        max_retries: Number of retry passes if series fails (rate-limited or transient).
        retry_wait: Seconds to wait between retry passes.
        silent: If True, suppress progress prints (useful when called from fetch_and_cache).

    Returns:
        DataFrame indexed by date (monthly or quarterly), columns named
        '{currency}_{indicator}' (e.g. 'USD_CPI', 'EUR_Policy Rate').
    """
    if series_map is None:
        series_map = DEFAULT_FRED_SERIES
    client = FREDClient()
    pending = dict(series_map)
    series_list: list[pd.Series] = []
    for attempt in range(max_retries):
        if not pending:
            break
        still_pending = {}
        for (currency, indicator), fred_id in pending.items():
            col = f"{currency}_{indicator}"
            try:
                s = client.get_series(fred_id, start=start, end=end)
                s.name = col
                series_list.append(s)
                if not silent:
                    print(f"  ✓ {col:35s} ({fred_id:30s}) n={len(s)}")
            except FREDRateLimitError as e:
                # Schedule for retry
                still_pending[(currency, indicator)] = fred_id
                if not silent:
                    print(f"  ⟳ {col:35s} rate-limited, will retry")
            except Exception as e:
                # Series doesn't exist or other permanent error
                if not silent:
                    print(f"  ✗ {col:35s} ({fred_id}): {str(e)[:50]}")
        pending = still_pending
        if pending and attempt < max_retries - 1:
            if not silent:
                print(f"  ⏳ Waiting {retry_wait}s before retry pass {attempt + 2}/{max_retries}...")
            time.sleep(retry_wait)
    if pending and not silent:
        for (currency, indicator) in pending:
            print(f"  ✗ {currency}_{indicator}: FAILED after {max_retries} attempts")
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1)
    df.index.name = "date"
    return df


def fetch_and_cache(
    cache_dir: Path | None = None,
    start: str = "2003-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch from FRED and write to CSV cache.

    Writes three files to `cache_dir`:
    - fred_panel_levels.csv (wide format, all series)
    - fred_panel_yoy.csv (wide format, YoY for index series)
    - fred_series_long.csv (long format: date, series, value)
    - fred_panel_levels.parquet (fast reload format)
    - fred_panel_yoy.parquet (fast reload format)
    - coverage.json

    If `force_refresh` is False and the CSV cache exists, returns the cached
    data without calling FRED. To force a FRED fetch, pass force_refresh=True.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "fred_panel_levels.csv"

    if csv_path.exists() and not force_refresh:
        print(f"Reading from cache: {csv_path}")
        return load_fred_panel(csv_path)

    print("Fetching from FRED (will be cached as CSV)...")
    panel = fetch_macro_panel(start=start, end=end, silent=False)
    if panel.empty:
        return panel

    # Ensure DatetimeIndex (parquet roundtrip can lose this)
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"

    # Wide-format CSVs (date as first column, no pandas index column)
    panel.reset_index().to_csv(
        cache_dir / "fred_panel_levels.csv", index=False, float_format="%.4f"
    )
    panel.to_parquet(cache_dir / "fred_panel_levels.parquet")
    print(f"  Saved wide-format CSV: {cache_dir / 'fred_panel_levels.csv'}")
    print(f"  Saved wide-format parquet: {cache_dir / 'fred_panel_levels.parquet'}")

    # YoY-transformed
    yoy_cols = [c for c in panel.columns if any(x in c for x in ["CPI", "PCE", "GDP"])]
    yoy_panel = pd.DataFrame(index=panel.index)
    for col in yoy_cols:
        periods = 4 if "GDP" in col else 12
        yoy_panel[col + "_YoY"] = compute_yoy(panel[col], periods=periods)
    if not yoy_panel.empty:
        if not isinstance(yoy_panel.index, pd.DatetimeIndex):
            yoy_panel.index = pd.to_datetime(yoy_panel.index)
        yoy_panel.index.name = "date"
        yoy_panel.reset_index().to_csv(
            cache_dir / "fred_panel_yoy.csv", index=False, float_format="%.4f"
        )
        yoy_panel.to_parquet(cache_dir / "fred_panel_yoy.parquet")
        print(f"  Saved YoY: {cache_dir / 'fred_panel_yoy.csv'}")

    # Long-format CSV
    panel_long = (
        panel.reset_index()
        .melt(id_vars="date", var_name="series", value_name="value")
        .dropna(subset=["value"])
        .sort_values(["series", "date"])
    )
    panel_long.to_csv(cache_dir / "fred_series_long.csv", index=False, float_format="%.4f")
    print(f"  Saved long-format CSV: {cache_dir / 'fred_series_long.csv'}")

    return panel


def load_fred_panel(csv_path: Path | str) -> pd.DataFrame:
    """Load a wide-format FRED panel CSV. First column is 'date'."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.set_index("date")


def load_fred_yoy(csv_path: Path | str) -> pd.DataFrame:
    """Load a wide-format FRED YoY panel CSV. First column is 'date'."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.set_index("date")


def load_fred_long(csv_path: Path | str) -> pd.DataFrame:
    """Load a long-format FRED series CSV. Columns: date, series, value."""
    return pd.read_csv(csv_path, parse_dates=["date"])


def fetch_indicator(
    client: FREDClient,
    currency: str,
    indicator: str,
    start: str = "2003-01-01",
    end: str | None = None,
) -> pd.Series:
    """Convenience: fetch one (currency, indicator) series."""
    fred_id = DEFAULT_FRED_SERIES[(currency, indicator)]
    return client.get_series(fred_id, start=start, end=end)


def compute_yoy(series: pd.Series, periods: int = 12) -> pd.Series:
    """Compute year-over-year % change of an index series."""
    return (series / series.shift(periods) - 1) * 100


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
