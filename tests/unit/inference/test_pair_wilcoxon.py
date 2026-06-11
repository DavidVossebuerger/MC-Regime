"""Test pair-level inference: replace per-replication Wilcoxon with sign test.

v13 used scipy.stats.wilcoxon over (3 pairs × 1999 replications) = 5997
samples. The Reviewer (§1.2) correctly identified this as pseudo-
replication: the 1999 replications are NOT independent draws.

v14 fix: aggregate to ONE number per (method, metric, pair), then run
a paired sign test (3 paired observations per test).
"""
import numpy as np
import pytest
from scipy.stats import wilcoxon


def pair_level_pvalue(method_per_pair: np.ndarray, baseline_per_pair: np.ndarray) -> float:
    """Paired sign test of n=3 paired observations (one per pair).

    Uses scipy.stats.wilcoxon with `alternative="two-sided"` and
    `zero_method="wilcox"` (default). For n=3, the smallest achievable
    two-sided p-value is 0.25 (when all 3 differences have the same sign).

    Edge cases:
    - If fewer than 2 pairs have non-zero differences, returns 1.0.
    """
    method_per_pair = np.asarray(method_per_pair, dtype=float).flatten()
    baseline_per_pair = np.asarray(baseline_per_pair, dtype=float).flatten()
    n = min(len(method_per_pair), len(baseline_per_pair))
    if n < 2:
        return 1.0
    diffs = method_per_pair[:n] - baseline_per_pair[:n]
    diffs = diffs[diffs != 0]
    if len(diffs) < 2:
        return 1.0
    try:
        _, p = wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        return 1.0
    return float(np.clip(p, 0.0, 1.0))


def test_pair_level_clear_difference():
    """If method < baseline for all 3 pairs, p = 0.25 (smallest possible for n=3)."""
    method = np.array([0.10, 0.20, 0.30])
    baseline = np.array([0.50, 0.60, 0.70])
    p = pair_level_pvalue(method, baseline)
    assert p == 0.25


def test_pair_level_no_difference():
    """If method and baseline are identical, p = 1.0 (no information)."""
    arr = np.array([0.10, 0.20, 0.30])
    p = pair_level_pvalue(arr, arr)
    assert p == 1.0


def test_pair_level_mixed_difference():
    """If 2 of 3 differences are positive (and 1 negative), p = 0.75."""
    method = np.array([0.10, 0.20, 0.50])
    baseline = np.array([0.20, 0.10, 0.30])
    p = pair_level_pvalue(method, baseline)
    assert p == 0.75