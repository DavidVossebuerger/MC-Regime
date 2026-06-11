"""Economic calendar loader for Forex Factory CSVs.

Reads the yearly forex_factory_calendar_*.csv files, parses numeric values
from Actual/Forecast/Previous fields (handles K/M/B/T/% suffixes and
range strings), computes surprises, and produces a unified DataFrame.

Output: pandas DataFrame with columns
    timestamp, currency, event, impact, actual, forecast, previous,
    actual_num, forecast_num, previous_num, surprise
where surprise = actual_num - forecast_num.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Numeric suffixes (multipliers)
SUFFIX_MAP = {
    "K": 1e3,
    "M": 1e6,
    "B": 1e9,
    "T": 1e12,
}

# Recognised impact levels
IMPACT_LEVELS = ["High", "Medium", "Low", "Non-economic"]


def _parse_numeric(value: object) -> float | None:
    """Parse a Forex Factory actual/forecast/previous string to a float.

    Handles:
    - Plain numbers: "47.4", "-0.4", "164"
    - Suffixes: "164K" -> 164000, "8.79M" -> 8_790_000, "1.2B" -> 1.2e9
    - Percent: "0.1%" -> 0.1 (kept as percentage points, not 0.001)
    - Ranges: "1.63|3.0" -> midpoint (2.315)
    - Empty / NaN / non-numeric strings -> None
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s == "-":
        return None
    # Range string: pick midpoint
    if "|" in s:
        parts = s.split("|")
        vals = [_parse_numeric(p) for p in parts]
        if all(v is not None for v in vals):
            return float(np.mean(vals))
        return None
    # Strip % (kept as percentage points for comparability)
    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()
    # Suffix
    if s and s[-1].upper() in SUFFIX_MAP:
        num_str = s[:-1].strip()
        mult = SUFFIX_MAP[s[-1].upper()]
    else:
        num_str = s
        mult = 1.0
    try:
        return float(num_str) * mult
    except (ValueError, TypeError):
        return None


def discover_calendar_files(calendar_dir: Path) -> list[Path]:
    """Find all forex_factory_calendar_*.csv files in calendar_dir, sorted by year."""
    files = sorted(calendar_dir.glob("forex_factory_calendar_*.csv"))
    if not files:
        raise FileNotFoundError(f"No calendar CSVs in {calendar_dir}")
    return files


def load_calendar(calendar_dir: Path) -> pd.DataFrame:
    """Load and concatenate all calendar CSVs.

    Returns DataFrame with columns:
        timestamp (UTC, pd.Timestamp), currency, event, impact,
        actual, forecast, previous (original strings),
        actual_num, forecast_num, previous_num (floats),
        surprise (actual_num - forecast_num, NaN if either missing)
    """
    files = discover_calendar_files(calendar_dir)
    dfs = []
    for path in files:
        df = pd.read_csv(path)
        # Normalise column names to lowercase for consistent access
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        dfs.append(df)
    cal = pd.concat(dfs, ignore_index=True)
    # Parse timestamp from combined_datetime
    cal["timestamp"] = pd.to_datetime(cal["combined_datetime"], utc=True, errors="coerce")
    # Drop rows without a valid timestamp
    cal = cal.dropna(subset=["timestamp"])
    # Drop duplicate events (same timestamp, currency, event, impact)
    cal = cal.drop_duplicates(
        subset=["timestamp", "currency", "event", "impact"], keep="last"
    )
    cal = cal.sort_values("timestamp").reset_index(drop=True)
    # Parse numerics
    cal["actual_num"] = cal["actual"].apply(_parse_numeric)
    cal["forecast_num"] = cal["forecast"].apply(_parse_numeric)
    cal["previous_num"] = cal["previous"].apply(_parse_numeric)
    # Surprise: actual - forecast
    cal["surprise"] = cal["actual_num"] - cal["forecast_num"]
    # Normalise impact column
    cal["impact"] = cal["impact"].astype(str).str.strip()
    # Keep only the columns we care about
    out = cal[[
        "timestamp", "currency", "event", "impact",
        "actual", "forecast", "previous",
        "actual_num", "forecast_num", "previous_num", "surprise",
    ]]
    return out


def filter_by_currency(cal: pd.DataFrame, currencies: list[str]) -> pd.DataFrame:
    """Filter calendar to specific currencies (e.g. ['USD', 'EUR', 'GBP', 'JPY'])."""
    return cal[cal["currency"].isin(currencies)].reset_index(drop=True)


def filter_by_impact(cal: pd.DataFrame, min_impact: str = "Medium") -> pd.DataFrame:
    """Filter calendar to events of at least `min_impact` importance.

    min_impact in {'High', 'Medium', 'Low', 'Non-economic'}.
    Higher importance (e.g. 'High') is a strict subset of lower ('Medium').
    """
    order = ["Non-economic", "Low", "Medium", "High"]
    if min_impact not in order:
        raise ValueError(f"min_impact must be one of {order}")
    threshold_idx = order.index(min_impact)
    allowed = order[threshold_idx:]
    return cal[cal["impact"].isin(allowed)].reset_index(drop=True)


def compute_surprise_index(
    cal: pd.DataFrame, currency: str, freq: str = "1D", standardize: bool = True
) -> pd.DataFrame:
    """Compute a daily/weekly surprise index for one currency.

    Steps:
    1. Filter calendar to `currency` events with available actual + forecast.
    2. Aggregate surprises within each time bin: sum (positive = net positive surprise).
    3. Standardise via rolling z-score (window = 60 days) to make comparable across events.

    Returns DataFrame indexed by timestamp, columns [surprise_raw, surprise_index].
    """
    sub = cal[
        (cal["currency"] == currency)
        & cal["surprise"].notna()
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["surprise_raw", "surprise_index"])
    # Many events are reported in different units (%, K, M, plain). Standardise per event type.
    # Group by event name and z-score within event type.
    sub["surprise_z_event"] = sub.groupby("event")["surprise"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )
    # Aggregate to time bins
    sub_indexed = sub.set_index("timestamp")
    agg = sub_indexed.resample(freq).agg(
        surprise_raw=("surprise_z_event", "sum"),
        n_events=("surprise_z_event", "count"),
    )
    agg = agg.fillna(0)
    if standardize:
        rolling_mean = agg["surprise_raw"].rolling(60, min_periods=10).mean()
        rolling_std = agg["surprise_raw"].rolling(60, min_periods=10).std()
        agg["surprise_index"] = (agg["surprise_raw"] - rolling_mean) / rolling_std
    else:
        agg["surprise_index"] = agg["surprise_raw"]
    return agg.reset_index()


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
