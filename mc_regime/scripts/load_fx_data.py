"""CLI: Load all FX pairs from raw Dukascopy CSVs, clean, save as Parquet.

Usage:
    python scripts/load_fx_data.py [--raw-dir DIR] [--out-dir DIR] [--freq FREQ]

Default:
    --raw-dir /root/research/MC-Regime    (contains eurusd-*.csv)
              + /root/research/MC-Regime/dukascopy_data (contains gbpusd-, usdjpy-)
    --out-dir  /root/research/MC-Regime/mc_regime/outputs/data
    --freq     1h (resample granularity for the output)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path so we can import mc_regime without install
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mc_regime.data.fx_loader import (
    clean_pair,
    inventory,
    load_pair,
    resample_bars,
    save_parquet,
)


# Pairs to process. CHFUSD is intentionally omitted (no data available).
DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]


def find_data_dirs(root: Path) -> dict[str, Path]:
    """Return a mapping of pair -> raw data directory.

    EURUSD lives in the project root; GBPUSD and USDJPY in dukascopy_data.
    """
    return {
        "EURUSD": root,  # eurusd-m30-bid-...csv
        "GBPUSD": root / "dukascopy_data",
        "USDJPY": root / "dukascopy_data",
    }


def quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame, df_resampled: pd.DataFrame, pair: str) -> dict:
    """Generate a quality report comparing raw / cleaned / resampled data."""
    raw = {
        "n_bars": int(len(df_raw)),
        "n_files": int(df_raw["pair"].count() > 0),
        "start": str(df_raw["timestamp"].min()),
        "end": str(df_raw["timestamp"].max()),
        "any_nan_ohlc": bool(df_raw[["open", "high", "low", "close"]].isna().any().any()),
        "any_negative_price": bool((df_raw[["open", "high", "low", "close"]] <= 0).any().any()),
        "duplicates": int(df_raw.duplicated(subset=["timestamp"]).sum()),
    }
    clean = {
        "n_bars": int(len(df_clean)),
        "start": str(df_clean["timestamp"].min()),
        "end": str(df_clean["timestamp"].max()),
        "n_weekend_dropped": int(len(df_raw) - len(df_clean)),
    }
    resampled = {
        "n_bars": int(len(df_resampled)),
        "start": str(df_resampled["timestamp"].min()),
        "end": str(df_resampled["timestamp"].max()),
    }
    return {"pair": pair, "raw": raw, "clean": clean, "resampled": resampled}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/root/research/MC-Regime")
    parser.add_argument("--out-dir", type=str, default="/root/research/MC-Regime/mc_regime/outputs/data")
    parser.add_argument("--freq", type=str, default="1h",
                        help="Output frequency (e.g., '1h', '4h', '1D', '1ME')")
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_PAIRS)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dirs = find_data_dirs(root)

    # 1. Inventory
    print("=" * 70)
    print("FX DATA LOADER — Dukascopy → Parquet")
    print("=" * 70)
    print(f"\nOutput directory: {out_dir}")
    print(f"Output frequency: {args.freq}")
    print(f"Pairs to process: {args.pairs}\n")

    print("--- INVENTORY ---")
    inv_combined = {}
    # Each pair maps to exactly one data dir; avoid double-scanning.
    for pair in args.pairs:
        if pair not in data_dirs:
            continue
        ddir = data_dirs[pair]
        if not ddir.exists():
            print(f"  {pair}: dir {ddir} does not exist — skipping")
            continue
        inv = inventory(ddir)
        if pair in inv:
            info = inv[pair]
            print(f"  {pair}: {info['n_files']} files, sample {info['n_bars_sample']} bars, "
                  f"{info['start_sample'].date()} → {info['end_sample'].date()}")
            inv_combined[pair] = info
        else:
            print(f"  {pair}: not found in {ddir}")

    missing = [p for p in args.pairs if p not in inv_combined and p in data_dirs]
    if missing:
        print(f"\nWARNING: No data found for pairs: {missing}")

    # 2. Load + clean + resample
    print("\n--- PROCESSING ---")
    all_reports = []
    for pair in args.pairs:
        if pair not in data_dirs:
            continue
        ddir = data_dirs[pair]
        if not ddir.exists():
            continue
        try:
            print(f"\n{pair}:")
            df_raw = load_pair(ddir, pair)
            print(f"  raw:        {len(df_raw):>10,} bars  ({df_raw['timestamp'].min().date()} → {df_raw['timestamp'].max().date()})")
            df_clean = clean_pair(df_raw, drop_weekends=True, drop_outliers=True)
            print(f"  cleaned:    {len(df_clean):>10,} bars  ({df_clean['timestamp'].min().date()} → {df_clean['timestamp'].max().date()})  "
                  f"(-{len(df_raw) - len(df_clean):,} dropped)")
            df_resampled = resample_bars(df_clean, freq=args.freq, agg="ohlc")
            print(f"  resampled:  {len(df_resampled):>10,} bars  @ {args.freq}  ({df_resampled['timestamp'].min().date()} → {df_resampled['timestamp'].max().date()})")

            # Save outputs
            save_parquet(df_clean, out_dir / f"{pair}_m30_clean.parquet")
            save_parquet(df_resampled, out_dir / f"{pair}_{args.freq}.parquet")
            print(f"  saved:      {out_dir / f'{pair}_m30_clean.parquet'}")
            print(f"  saved:      {out_dir / f'{pair}_{args.freq}.parquet'}")

            report = quality_report(df_raw, df_clean, df_resampled, pair)
            all_reports.append(report)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
        except Exception as e:
            print(f"  ERROR processing {pair}: {e}")
            raise

    # 3. Save quality report
    report_path = out_dir / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n--- QUALITY REPORT ---")
    print(f"Saved: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
