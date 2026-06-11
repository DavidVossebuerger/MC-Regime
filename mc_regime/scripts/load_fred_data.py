"""CLI: Fetch macroeconomic panel from FRED, cache as CSV.

Default behaviour: read from local CSV cache. Use --refresh to re-fetch
from FRED (rate-limited) and overwrite the cache.

Usage:
    python scripts/load_fred_data.py                    # read from cache
    python scripts/load_fred_data.py --refresh          # re-fetch from FRED
    python scripts/load_fred_data.py --start 2010-01-01 # custom start
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from mc_regime.data.fred_loader import (
    DEFAULT_CACHE_DIR,
    compute_yoy,
    fetch_and_cache,
    fetch_macro_panel,
    load_fred_long,
    load_fred_panel,
    load_fred_yoy,
    save_parquet,
)
from mc_regime.data.macro_loader import macro_wide_table

DEFAULT_CALENDAR_DIR = Path("/root/research/MC-Regime/Economic-Calendar")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2003-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--calendar-dir", type=str, default=str(DEFAULT_CALENDAR_DIR))
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch from FRED (overwrites cache)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FRED MACRO LOADER — CSV cache (FRED API only on --refresh)")
    print("=" * 70)
    print(f"\nOutput: {out_dir}")
    print(f"Date range: {args.start} → {args.end or 'latest'}")
    print(f"Mode: {'REFRESH from FRED' if args.refresh else 'read from cache'}\n")

    # 1. Fetch or read from cache
    if args.refresh:
        panel = fetch_and_cache(
            cache_dir=out_dir, start=args.start, end=args.end, force_refresh=True
        )
    else:
        csv_path = out_dir / "fred_panel_levels.csv"
        if not csv_path.exists():
            print(f"  Cache not found: {csv_path}")
            print("  Running fetch from FRED to build cache...")
            panel = fetch_and_cache(
                cache_dir=out_dir, start=args.start, end=args.end, force_refresh=True
            )
        else:
            print(f"Reading from cache: {csv_path}")
            panel = load_fred_panel(csv_path)

    if panel.empty:
        print("  No data available!")
        return
    print(f"\nLoaded: {panel.shape[0]} dates × {panel.shape[1]} series")
    print(f"Date range: {panel.index.min().date()} → {panel.index.max().date()}")

    # 2. Load YoY
    yoy_path = out_dir / "fred_panel_yoy.csv"
    if yoy_path.exists():
        yoy = load_fred_yoy(yoy_path)
    else:
        # Compute on the fly from panel
        yoy_cols = [c for c in panel.columns if any(x in c for x in ["CPI", "PCE", "GDP"])]
        yoy = pd.DataFrame(index=panel.index)
        for col in yoy_cols:
            periods = 4 if "GDP" in col else 12
            yoy[col + "_YoY"] = compute_yoy(panel[col], periods=periods)

    # 3. Coverage summary
    print("\n--- COVERAGE ---")
    coverage = {}
    for col in panel.columns:
        n = int(panel[col].notna().sum())
        if n > 0:
            coverage[col] = {
                "n_obs": n,
                "first": str(panel[col].dropna().index.min().date()),
                "last": str(panel[col].dropna().index.max().date()),
            }
            print(f"  {col:35s}  n={n:>5}  {coverage[col]['first']} → {coverage[col]['last']}")
    with open(out_dir / "coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    # 4. Comparison with Calendar
    print("\n--- COMPARISON: FRED vs Calendar ---")
    calendar_wide = macro_wide_table(Path(args.calendar_dir))
    if calendar_wide.empty:
        print("  No calendar data to compare")
    else:
        comparison = {}
        for col in panel.columns:
            ccy, indicator = col.split("_", 1)
            cal_col = None
            for c in calendar_wide.columns:
                if c.startswith(ccy + "_"):
                    event = c[len(ccy) + 1:].lower()
                    if indicator.lower().split()[0] in event:
                        cal_col = c
                        break
            fred_n = coverage.get(col, {}).get("n_obs", 0)
            cal_n = int(calendar_wide[cal_col].notna().sum()) if cal_col else 0
            comparison[col] = {
                "fred_obs": fred_n,
                "calendar_obs": cal_n,
                "calendar_col": cal_col,
            }
            marker = "  ✓" if cal_n > 0 else "  -"
            print(f"{marker} {col:35s}  FRED={fred_n:>5}  Calendar={cal_n:>4}  ({cal_col or 'no match'})")
        with open(out_dir / "comparison_calendar.json", "w") as f:
            json.dump(comparison, f, indent=2)

    # 5. Sample
    print("\n--- SAMPLE (last 5 rows of levels, top 6 series) ---")
    top_cols = sorted(coverage.keys(), key=lambda c: coverage[c]["n_obs"], reverse=True)[:6]
    print(panel[top_cols].tail(5).round(2).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
