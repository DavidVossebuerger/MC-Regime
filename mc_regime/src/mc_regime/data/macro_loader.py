"""Macro time-series loader for Forex Factory calendar CSVs.

Extracts per-event-type time series of Actual values from the calendar.
For example, 'CPI y/y' for USD gives a monthly time series of US CPI
year-over-year readings.

Output: dict of (currency, event) -> DataFrame with columns
    timestamp, value, value_num
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mc_regime.data.calendar_loader import _parse_numeric, load_calendar


# Standard set of macro events that produce monthly time series.
# Used as a default for the macro_extractor; users can override.
DEFAULT_EVENTS = {
    "USD": [
        "CPI y/y", "Core CPI m/m", "Unemployment Rate",
        "Non-Farm Employment Change", "ISM Manufacturing PMI",
        "GDP m/m", "PPI m/m", "Core Retail Sales m/m",
        "Retail Sales m/m", "Core PCE Price Index m/m",
    ],
    "EUR": [
        "CPI Flash Estimate y/y", "Core CPI Flash Estimate y/y",
        "Main Refinancing Rate", "German Prelim CPI m/m",
        "Flash Manufacturing PMI", "Flash Services PMI",
        "German ifo Business Climate",
    ],
    "GBP": [
        "CPI y/y", "Core CPI y/y", "Unemployment Rate",
        "Claimant Count Change", "MPC Official Bank Rate Votes",
        "Manufacturing PMI", "Services PMI", "GDP m/m",
    ],
    "JPY": [
        "National Core CPI y/y", "BoJ Policy Rate", "Unemployment Rate",
        "Industrial Production m/m", "Manufacturing PMI", "Services PMI",
    ],
    "CHF": [
        "CPI m/m", "CPI y/y", "SNB Policy Rate", "Unemployment Rate",
        "Manufacturing PMI",
    ],
}


def extract_macro_series(
    calendar: pd.DataFrame,
    currency: str,
    event: str,
    min_impact: str = "Medium",
) -> pd.DataFrame:
    """Extract a single (currency, event) time series from the calendar.

    Returns DataFrame with columns [timestamp, actual_str, value].
    `value` is the parsed numeric Actual; events without a numeric Actual
    are dropped.
    """
    sub = calendar[
        (calendar["currency"] == currency)
        & (calendar["event"] == event)
    ].copy()
    if min_impact:
        order = ["Non-economic", "Low", "Medium", "High"]
        threshold_idx = order.index(min_impact)
        allowed = order[threshold_idx:]
        sub = sub[sub["impact"].isin(allowed)]
    if sub.empty:
        return pd.DataFrame(columns=["timestamp", "actual_str", "value"])
    sub["value"] = sub["actual_num"]
    out = sub[["timestamp", "actual", "value"]].rename(columns={"actual": "actual_str"})
    out = out.dropna(subset=["value"])
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def extract_all_macro(
    calendar_dir: Path,
    events: dict[str, list[str]] | None = None,
    min_impact: str = "Medium",
) -> dict[tuple[str, str], pd.DataFrame]:
    """Extract time series for all (currency, event) pairs in `events`.

    Returns dict {(currency, event): DataFrame}.
    """
    if events is None:
        events = DEFAULT_EVENTS
    cal = load_calendar(calendar_dir)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for currency, event_list in events.items():
        for event in event_list:
            ts = extract_macro_series(cal, currency, event, min_impact=min_impact)
            if not ts.empty:
                out[(currency, event)] = ts
    return out


def macro_wide_table(
    calendar_dir: Path,
    events: dict[str, list[str]] | None = None,
    min_impact: str = "Medium",
) -> pd.DataFrame:
    """Build a wide-format DataFrame: index = month-end timestamp,
    columns = (currency, event) tuples flattened to 'CCY_EVENT' strings.

    Useful for HMM input where we need a single DataFrame with all
    macro features as columns.
    """
    if events is None:
        events = DEFAULT_EVENTS
    cal = load_calendar(calendar_dir)
    series_list = []
    for currency, event_list in events.items():
        for event in event_list:
            ts = extract_macro_series(cal, currency, event, min_impact=min_impact)
            if ts.empty:
                continue
            # Floor timestamp to month-end; aggregate per month (last wins)
            ts["month"] = ts["timestamp"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")
            col_name = f"{currency}_{event}"
            ts_renamed = ts.rename(columns={"value": col_name})
            # Per month, take the last value (or mean if multiple releases in same month)
            ts_agg = ts_renamed.groupby("month", as_index=False)[col_name].last()
            series_list.append(ts_agg)
    if not series_list:
        return pd.DataFrame()
    # Merge all on month
    result = series_list[0]
    for s in series_list[1:]:
        result = result.merge(s, on="month", how="outer")
    result = result.sort_values("month").reset_index(drop=True)
    result = result.rename(columns={"month": "timestamp"})
    return result


def save_macro_wide(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_macro_wide(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
