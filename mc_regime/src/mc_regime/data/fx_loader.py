"""FX data loader for Dukascopy CSV files.

Reads all CSV files for a given currency pair, deduplicates, filters
out indices, and produces a unified DataFrame with columns:
    timestamp, open, high, low, close, pair
where timestamp is a UTC datetime.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Files matching these patterns are NOT FX pairs (they are indices).
INDEX_PATTERNS = ("usa500idxusd", "usatechidxusd", "idxusd", "nasdaq")

# Recognized pair prefixes in Dukascopy filenames.
# USD/JPY is quoted as "1 USD = X JPY" (FX broker standard), not "1 JPY = X USD".
KNOWN_PAIRS = {
    "eurusd": "EURUSD",
    "gbpusd": "GBPUSD",
    "usdjpy": "USDJPY",
    "audusd": "AUDUSD",
    "nzdusd": "NZDUSD",
    "usdcad": "USDCAD",
    "usdchf": "USDCHF",
    "eurjpy": "EURJPY",
    "eurgbp": "EURGBP",
    "eurchf": "EURCHF",
    "gbpjpy": "GBPJPY",
}


def _is_index_file(path: Path) -> bool:
    name = path.name.lower()
    return any(pat in name for pat in INDEX_PATTERNS)


def _pair_from_filename(path: Path) -> str | None:
    """Extract pair code from a Dukascopy-style filename. Returns None if not a known FX pair."""
    name = path.stem.lower()
    # Strip frequency and date suffix variants
    # e.g. "gbpusd-m30-bid-2007-01-01-2007-03-31" or "gbpusd_2007-q1"
    stem = re.split(r"[-_]", name)[0]
    return KNOWN_PAIRS.get(stem)


def discover_files(data_dir: Path, pair_code: str) -> list[Path]:
    """Find all CSV files for a given pair (case-insensitive) in data_dir.

    pair_code: e.g. 'EURUSD' or 'eurusd' or 'USDJPY'.
    Returns: sorted list of Paths (duplicates with different naming may appear).
    """
    pair_lower = pair_code.lower()
    matches: list[Path] = []
    for path in sorted(data_dir.glob("*.csv")):
        if _is_index_file(path):
            continue
        # Accept file if its parsed pair matches OR filename starts with pair code
        parsed = _pair_from_filename(path)
        if parsed == pair_code.upper() or path.stem.lower().startswith(pair_lower):
            matches.append(path)
    return matches


def _read_single_csv(path: Path) -> pd.DataFrame:
    """Read a Dukascopy CSV, return DataFrame with parsed timestamps."""
    df = pd.read_csv(path)
    # Schema: timestamp, open, high, low, close (and possibly volume)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close"]]


def load_pair(data_dir: Path, pair_code: str) -> pd.DataFrame:
    """Load and concatenate all CSV files for a given pair.

    Handles multiple naming conventions and overlapping quarter files.
    Deduplicates on timestamp.
    """
    files = discover_files(data_dir, pair_code)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found for pair {pair_code} in {data_dir}. "
            f"Known pairs: {list(KNOWN_PAIRS.values())}"
        )
    dfs = [_read_single_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate on timestamp (last write wins, since later files have more recent data)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["pair"] = pair_code.upper()
    return combined


def clean_pair(
    df: pd.DataFrame,
    drop_weekends: bool = True,
    drop_outliers: bool = True,
    outlier_std: float = 8.0,
) -> pd.DataFrame:
    """Clean a pair DataFrame: NaN, duplicates, weekends, outliers.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, pair].
        drop_weekends: Remove Saturday/Sunday (FX is 24/5, but Dukascopy includes gaps).
        drop_outliers: Remove bars whose return exceeds outlier_std × rolling std.
        outlier_std: Z-score threshold for outlier removal.
    """
    df = df.copy()
    # 1. Drop NaN in OHLC
    df = df.dropna(subset=["open", "high", "low", "close"])
    # 2. Drop zero or negative prices
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    # 3. Drop weekends (Saturday=5, Sunday=6)
    if drop_weekends:
        df = df[df["timestamp"].dt.weekday < 5]
    # 4. Outlier detection: extreme returns
    if drop_outliers and len(df) > 100:
        df["_logret"] = np.log(df["close"] / df["close"].shift(1))
        rolling_std = df["_logret"].rolling(window=500, min_periods=50).std()
        df["_z"] = (df["_logret"] - df["_logret"].rolling(500, min_periods=50).mean()) / rolling_std
        df = df[df["_z"].abs() < outlier_std]
        df = df.drop(columns=["_logret", "_z"])
    df = df.reset_index(drop=True)
    return df


def resample_bars(
    df: pd.DataFrame, freq: str = "1h", agg: str = "ohlc"
) -> pd.DataFrame:
    """Resample a 30-min bar DataFrame to a higher frequency.

    Args:
        df: DataFrame with [timestamp, open, high, low, close, pair].
        freq: pandas frequency string ('1h', '4h', '1D', '1W', '1ME' for month-end).
        agg: 'ohlc' for proper OHLC aggregation, 'last' for close only.
    """
    df = df.set_index("timestamp")
    if agg == "ohlc":
        out = df.resample(freq).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "pair": "first"}
        )
    elif agg == "last":
        out = df.resample(freq).agg({"close": "last", "pair": "first"})
        out = out.rename(columns={"close": "FX"})
    else:
        raise ValueError(f"Unknown agg: {agg}")
    out = out.dropna(subset=["pair"]).reset_index()
    return out


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet."""
    return pd.read_parquet(path)


def inventory(data_dir: Path) -> dict[str, dict]:
    """Return inventory of all FX pairs found in data_dir.

    Returns: {pair_code: {'files': [paths], 'n_bars': int, 'start': ts, 'end': ts}}
    """
    inventory: dict[str, dict] = {}
    for path in sorted(data_dir.glob("*.csv")):
        if _is_index_file(path):
            continue
        parsed = _pair_from_filename(path)
        if parsed is None:
            continue
        inv = inventory.setdefault(parsed, {"files": []})
        inv["files"].append(path)
    for code, inv in inventory.items():
        # Quick scan of one file to get row count
        sample = _read_single_csv(inv["files"][0])
        inv["n_files"] = len(inv["files"])
        inv["n_bars_sample"] = len(sample)
        inv["start_sample"] = sample["timestamp"].min()
        inv["end_sample"] = sample["timestamp"].max()
    return inventory
