"""CLI: Extract macro time series from Forex Factory calendar CSVs.

For each (currency, event) pair, builds a monthly time series of Actual
values, then merges all into a single wide table for HMM input.

Usage:
    python scripts/load_macro_series.py [--calendar-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from mc_regime.data.macro_loader import (
    DEFAULT_EVENTS,
    extract_macro_series,
    load_calendar,
    macro_wide_table,
    save_macro_wide,
)

DEFAULT_CALENDAR_DIR = Path("/root/research/MC-Regime/Economic-Calendar")
DEFAULT_OUT_DIR = Path("/root/research/MC-Regime/mc_regime/outputs/data/macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar-dir", type=str, default=str(DEFAULT_CALENDAR_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-impact", type=str, default="Medium")
    args = parser.parse_args()

    cal_dir = Path(args.calendar_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MACRO TIME-SERIES LOADER — Forex Factory Actual values → Parquet")
    print("=" * 70)
    print(f"\nSource: {cal_dir}")
    print(f"Output: {out_dir}")
    print(f"Min impact: {args.min_impact}\n")

    # 1. Per (currency, event) coverage
    print("--- PER-EVENT COVERAGE ---")
    cal = load_calendar(cal_dir)
    coverage = {}
    for currency, event_list in DEFAULT_EVENTS.items():
        for event in event_list:
            ts = extract_macro_series(cal, currency, event, min_impact=args.min_impact)
            key = f"{currency}_{event}"
            if ts.empty:
                coverage[key] = {"n_obs": 0}
                continue
            coverage[key] = {
                "n_obs": int(len(ts)),
                "first": str(ts["timestamp"].min().date()),
                "last": str(ts["timestamp"].max().date()),
                "mean": float(ts["value"].mean()) if ts["value"].notna().any() else None,
            }
    # Print per currency
    for currency in DEFAULT_EVENTS:
        print(f"\n  {currency}:")
        for event in DEFAULT_EVENTS[currency]:
            key = f"{currency}_{event}"
            cov = coverage[key]
            if cov["n_obs"] == 0:
                print(f"    {event:40s}  — no data")
            else:
                print(f"    {event:40s}  n={cov['n_obs']:>3}  "
                      f"{cov['first']} → {cov['last']}  "
                      f"mean={cov['mean']:.3f}" if cov['mean'] is not None else "")

    # 2. Build wide table
    print("\n--- WIDE TABLE ---")
    wide = macro_wide_table(cal_dir, min_impact=args.min_impact)
    if wide.empty:
        print("  ERROR: wide table is empty")
        return
    n_months = len(wide)
    n_cols = wide.shape[1] - 1
    print(f"  Shape: {wide.shape[0]} months × {wide.shape[1]} columns ({n_cols} macro series)")
    print(f"  Date range: {wide['timestamp'].min().date()} → {wide['timestamp'].max().date()}")
    # Coverage summary per column
    coverage_pct = (wide.iloc[:, 1:].notna().mean() * 100).round(1)
    print(f"  Coverage: median {coverage_pct.median():.1f}%, "
          f"min {coverage_pct.min():.1f}%, max {coverage_pct.max():.1f}%")
    save_macro_wide(wide, out_dir / "macro_wide.parquet")
    print(f"  Saved: {out_dir / 'macro_wide.parquet'}")

    # 3. Save coverage report
    with open(out_dir / "coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"\n  Coverage report: {out_dir / 'coverage.json'}")

    # 4. Sample output
    print("\n--- SAMPLE (last 5 months) ---")
    sample = wide.tail(5).copy()
    # Show timestamp + 4 most-covered columns
    top_cols = coverage_pct.nlargest(4).index.tolist()
    cols_to_show = ["timestamp"] + top_cols
    print(sample[cols_to_show].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
