"""Level 2: Wasserstein distance metrics.

This module computes Wasserstein-1 (Earth Mover's) distances between
empirical bootstrap distributions and point masses at original values.
"""
from __future__ import annotations

import numpy as np


def compute_wasserstein_distances(
    original: dict[tuple[str, str], float],
    bootstrap_distributions: list[dict[tuple[str, str], float]],
) -> dict[tuple[str, str], float]:
    """Compute Wasserstein-1 distance for each correlation pair.

    W1 = mean(|boot_values - original_value|).
    """
    if not bootstrap_distributions:
        raise ValueError("bootstrap_distributions cannot be empty")

    result = {}
    for pair in original.keys():
        if pair not in original:
            continue
        orig_val = original[pair]

        boot_values = np.array([dist.get(pair, 0.0) for dist in bootstrap_distributions])
        # Filter NaN
        valid_vals = boot_values[~np.isnan(boot_values)]
        if len(valid_vals) == 0:
            result[pair] = 0.0
        else:
            w1 = np.mean(np.abs(valid_vals - orig_val))
            result[pair] = w1

    return result


def compute_l2_wasserstein(
    bootstrap_distributions: list[dict[tuple[str, str], float]],
    original: dict[tuple[str, str], float] | None = None,
) -> float:
    """Compute Level 2: average Wasserstein-1 distance across all pairs."""
    if not bootstrap_distributions:
        raise ValueError("bootstrap_distributions cannot be empty")

    if original is None:
        original = bootstrap_distributions[0]
        distributions = bootstrap_distributions[1:]
    else:
        distributions = bootstrap_distributions

    w_distances = compute_wasserstein_distances(original, distributions)

    # L2 = mean of W1 distances (filter NaN)
    valid_dist = [d for d in w_distances.values() if not np.isnan(d)]
    if not valid_dist:
        return 0.0
    return np.mean(valid_dist)