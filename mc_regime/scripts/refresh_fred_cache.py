"""Refresh the FRED panel cache by adding missing series.

This script:
  1. Loads existing cache from fred_panel_levels.csv
  2. Identifies missing columns from DEFAULT_FRED_SERIES
  3. Fetches only the missing ones from FRED API
  4. Merges into existing CSV (preserves existing data)
  5. Regenerates YoY + long-format caches
  6. Prints summary of added/refreshed series

Usage:
    python scripts/refresh_fred_cache.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from mc_regime.data.fred_loader import (
    DEFAULT_FRED_SERIES,
    FREDClient,
    FREDRateLimitError,
    compute_yoy,
    DEFAULT_CACHE_DIR,
)


def find_missing_columns(existing_df: pd.DataFrame) -> list[tuple]:
    """Identify which (currency, indicator) pairs from DEFAULT_FRED_SERIES are missing."""
    existing_cols = set(existing_df.columns)
    missing = []

    for (curr, ind), fred_id in DEFAULT_FRED_SERIES.items():
        col_name = f"{curr}_{ind}"
        if col_name not in existing_cols:
            missing.append(((curr, ind), fred_id, col_name))

    return missing


def fetch_missing_series(
    missing: list[tuple],
    start: str = "2003-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch just the missing series from FRED."""
    if not missing:
        return pd.DataFrame()

    client = FREDClient()
    fetched = {}

    for (curr, ind), fred_id, col_name in missing:
        print(f"  Fetching {col_name} ({fred_id})...")
        try:
            s = client.get_series(fred_id, start=start, end=end)
            s.name = col_name
            fetched[col_name] = s
            print(f"    ✓ Got {len(s)} observations")
        except FREDRateLimitError as e:
            print(f"    ! Rate limited: {e}")
            continue  # Skip this series, try next
        except Exception as e:
            print(f"    ! Skipping ({type(e).__name__}): {e}")
            continue  # Skip failed series

    if not fetched:
        return pd.DataFrame()

    return pd.concat(fetched.values(), axis=1)


def merge_and_save(
    existing: pd.DataFrame,
    new_cols: pd.DataFrame,
    cache_dir: Path,
) -> None:
    """Merge new columns into existing panel and regenerate all caches."""
    if new_cols.empty:
        print("No new columns to merge.")
        return

    # Align indices - forward-fill missing dates where needed
    combined = existing.join(new_cols, how="outer")

    # Write wide-format CSV
    combined.reset_index().to_csv(
        cache_dir / "fred_panel_levels.csv",
        index=False,
        float_format="%.4f",
    )
    combined.to_parquet(cache_dir / "fred_panel_levels.parquet")
    print(f"  Saved: {cache_dir / 'fred_panel_levels.csv'}")

    # YoY-transformed columns
    yoy_cols = [c for c in combined.columns if any(x in c for x in ["CPI", "PCE", "GDP"])]
    yoy_panel = pd.DataFrame(index=combined.index)
    for col in yoy_cols:
        periods = 4 if "GDP" in col else 12
        yoy_panel[col + "_YoY"] = compute_yoy(combined[col], periods=periods)

    if not yoy_panel.empty:
        yoy_panel.reset_index().to_csv(
            cache_dir / "fred_panel_yoy.csv",
            index=False,
            float_format="%.4f",
        )
        yoy_panel.to_parquet(cache_dir / "fred_panel_yoy.parquet")
        print(f"  Saved: {cache_dir / 'fred_panel_yoy.csv'}")

    # Long-format CSV
    long_df = (
        combined.reset_index()
        .melt(id_vars="date", var_name="series", value_name="value")
        .dropna(subset=["value"])
        .sort_values(["series", "date"])
    )
    long_df.to_csv(cache_dir / "fred_series_long.csv", index=False, float_format="%.4f")
    print(f"  Saved: {cache_dir / 'fred_series_long.csv'}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Refresh FRED panel cache with missing series")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show missing columns, don't fetch or write",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing fred_panel_levels.csv",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    csv_path = cache_dir / "fred_panel_levels.csv"

    if not csv_path.exists():
        print(f"ERROR: No existing cache at {csv_path}")
        sys.exit(1)

    # Load existing cache
    print(f"Loading existing cache: {csv_path}")
    existing = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    print(f"  Shape: {existing.shape}")

    # Find missing
    missing = find_missing_columns(existing)
    print(f"\nFound {len(missing)} missing columns:")
    for (curr, ind), fred_id, col_name in missing:
        print(f"  - {col_name} ({fred_id})")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    if not missing:
        print("\nNo missing columns to fetch.")
        return

    # Fetch missing
    print("\nFetching missing series from FRED...")
    try:
        new_cols = fetch_missing_series(missing)
    except FREDRateLimitError:
        print("WARNING: Both FRED API keys rate-limited. No new series fetched.")
        new_cols = pd.DataFrame()

    if new_cols.empty:
        print("\nNo series could be fetched.")
        sys.exit(1)

    # Merge and save
    print("\nMerging into existing cache...")
    merged = merge_and_save(existing, new_cols, cache_dir)

    print(f"\nDone. New shape: {merged.shape}")


if __name__ == "__main__":
    main()