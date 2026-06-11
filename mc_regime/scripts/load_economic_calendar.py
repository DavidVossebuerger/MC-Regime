"""CLI: Process Forex Factory economic calendar CSVs.

Loads all yearly CSVs, parses numerics, computes surprise index per currency,
and saves everything as parquet.

Usage:
    python scripts/load_economic_calendar.py [--calendar-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from mc_regime.data.calendar_loader import (
    compute_surprise_index,
    filter_by_currency,
    filter_by_impact,
    load_calendar,
    save_parquet,
)

DEFAULT_CALENDAR_DIR = Path("/root/research/MC-Regime/Economic-Calendar")
DEFAULT_OUT_DIR = Path("/root/research/MC-Regime/mc_regime/outputs/data/calendar")
DEFAULT_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar-dir", type=str, default=str(DEFAULT_CALENDAR_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--currencies", type=str, nargs="+", default=DEFAULT_CURRENCIES)
    parser.add_argument("--min-impact", type=str, default="Medium",
                        choices=["Non-economic", "Low", "Medium", "High"])
    args = parser.parse_args()

    cal_dir = Path(args.calendar_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ECONOMIC CALENDAR LOADER — Forex Factory → Parquet")
    print("=" * 70)
    print(f"\nSource: {cal_dir}")
    print(f"Output: {out_dir}")
    print(f"Currencies: {args.currencies}")
    print(f"Min impact: {args.min_impact}\n")

    # 1. Load all
    print("--- LOADING ---")
    cal = load_calendar(cal_dir)
    print(f"  Total events loaded: {len(cal):,}")
    print(f"  Date range: {cal['timestamp'].min().date()} → {cal['timestamp'].max().date()}")
    print(f"  Currencies present: {sorted(cal['currency'].unique())}")
    print(f"  Impact distribution:")
    for imp, n in cal["impact"].value_counts().items():
        print(f"    {imp:15s}: {n:>6,}")

    # 2. Save full calendar
    save_parquet(cal, out_dir / "calendar_all.parquet")
    print(f"\n  Saved: {out_dir / 'calendar_all.parquet'}")

    # 3. Filter for our currencies + impact
    cal_filtered = filter_by_currency(cal, args.currencies)
    cal_filtered = filter_by_impact(cal_filtered, args.min_impact)
    print(f"\n  After filter ({', '.join(args.currencies)}, impact>={args.min_impact}): {len(cal_filtered):,}")
    save_parquet(cal_filtered, out_dir / "calendar_filtered.parquet")
    print(f"  Saved: {out_dir / 'calendar_filtered.parquet'}")

    # 4. Surprise index per currency
    print("\n--- SURPRISE INDEX ---")
    summary = {}
    for currency in args.currencies:
        idx = compute_surprise_index(cal, currency, freq="1D", standardize=True)
        if idx.empty:
            print(f"  {currency}: no events with actual+forecast")
            continue
        n_days = (idx["surprise_index"].notna()).sum()
        n_events = int(idx["n_events"].sum())
        mean_idx = idx["surprise_index"].mean()
        std_idx = idx["surprise_index"].std()
        # Coverage
        first = idx["timestamp"].min().date()
        last = idx["timestamp"].max().date()
        print(f"  {currency}: {n_events:>5,} events, {n_days:>5,} days, "
              f"mean idx = {mean_idx:+.3f}, std = {std_idx:.3f}  ({first} → {last})")
        save_parquet(idx, out_dir / f"surprise_index_{currency}.parquet")
        summary[currency] = {
            "n_events": n_events,
            "n_days": int(n_days),
            "first": str(first),
            "last": str(last),
            "mean_idx": float(mean_idx),
            "std_idx": float(std_idx),
        }

    # 5. Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # 6. Per-event-type summary for high-impact events
    print("\n--- HIGH-IMPACT EVENT COVERAGE ---")
    high = cal[(cal["impact"] == "High") & (cal["currency"].isin(args.currencies))]
    high_with_data = high[high["actual_num"].notna() & high["forecast_num"].notna()]
    by_event = high_with_data.groupby("event").agg(
        n=("timestamp", "count"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
    ).sort_values("n", ascending=False).head(20)
    for event, row in by_event.iterrows():
        first = row["first"].date()
        last = row["last"].date()
        print(f"  {event:40s}  n={int(row['n']):>4}  {first} → {last}")

    print("\nDone.")


if __name__ == "__main__":
    main()
