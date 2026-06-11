"""Level 1: Correlation bias metrics.

This module computes Pearson correlation coefficients between variable pairs
and measures bias between original and bootstrap distributions.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Default correlation pairs for the regime-preserving bootstrap.
DEFAULT_CORRELATION_PAIRS = [
    ("FX", "RateDiff"),
    ("FX", "InflationDiff"),
    ("FX", "Surprise"),
    ("FX", "Carry"),
    ("FX", "FVG"),
    ("FX", "RealVol"),
    ("RateDiff", "InflationDiff"),
    ("RateDiff", "Carry"),
    ("InflationDiff", "Carry"),
]

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_correlations(
    data: pd.DataFrame,
    pairs: "Sequence[tuple[str, str]]" | None = None,
) -> dict[tuple[str, str], float]:
    """Compute Pearson correlations for all defined variable pairs."""
    if pairs is None:
        pairs = DEFAULT_CORRELATION_PAIRS

    result = {}
    for v1, v2 in pairs:
        if v1 not in data.columns or v2 not in data.columns:
            warnings.warn(f"Skipping undefined pair ({v1}, {v2})")
            continue

        # Compute Pearson correlation (handle NaN)
        valid_mask = ~(data[v1].isna() | data[v2].isna())
        if valid_mask.sum() < 2:
            warnings.warn(f"Insufficient data for pair ({v1}, {v2})")
            result[(v1, v2)] = 0.0
        else:
            corr = data.loc[valid_mask, v1].corr(data.loc[valid_mask, v2])
            if np.isnan(corr):
                corr = 0.0
            result[(v1, v2)] = corr

    return result


def compute_correlation_bias(
    original: dict[tuple[str, str], float],
    bootstrap_distributions: list[dict[tuple[str, str], float]],
) -> dict[tuple[str, str], float]:
    """Compute bias: mean(boot) - original for each pair."""
    if not bootstrap_distributions:
        raise ValueError("bootstrap_distributions cannot be empty")

    pairs = list(original.keys())
    result = {}
    for pair in pairs:
        if pair not in original:
            continue
        boot_values = [dist.get(pair, 0.0) for dist in bootstrap_distributions]
        mean_boot = np.nanmean(boot_values)
        bias = mean_boot - original[pair]
        result[pair] = bias

    return result


def compute_l1_bias(
    bootstrap_distributions: list[dict[tuple[str, str], float]],
    original: dict[tuple[str, str], float] | None = None,
) -> float:
    """Compute Level 1 bias: average absolute bias across all correlation pairs."""
    if not bootstrap_distributions:
        raise ValueError("bootstrap_distributions cannot be empty")

    if original is None:
        original = bootstrap_distributions[0]
        distributions = bootstrap_distributions[1:]
    else:
        distributions = bootstrap_distributions

    biases = compute_correlation_bias(original, distributions)

    # L1 = mean of absolute biases (filter NaN)
    valid_biases = [abs(b) for b in biases.values() if not np.isnan(b)]
    if not valid_biases:
        return 0.0
    return np.mean(valid_biases)