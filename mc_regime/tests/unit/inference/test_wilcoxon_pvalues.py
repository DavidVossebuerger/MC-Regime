"""Test real Wilcoxon p-value computation (replaces v11's fake p-values)."""
import numpy as np
import pytest
from scipy.stats import wilcoxon


def compute_wilcoxon_pvalue(method_dist: np.ndarray, baseline_dist: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test of paired (method, baseline) loss distributions.

    Returns the two-sided p-value that the median of the paired differences
    is zero. Small p-values indicate the methods produce different loss distributions.

    Edge cases:
    - If either array is empty or constant, returns 1.0 (no information).
    - Drops exact-zero differences before the test.
    """
    method_dist = np.asarray(method_dist, dtype=float).flatten()
    baseline_dist = np.asarray(baseline_dist, dtype=float).flatten()

    n = min(len(method_dist), len(baseline_dist))
    if n < 2:
        return 1.0
    if method_dist.std() == 0 and baseline_dist.std() == 0:
        return 1.0

    diffs = method_dist[:n] - baseline_dist[:n]
    diffs = diffs[diffs != 0]
    if len(diffs) < 2:
        return 1.0

    _, p = wilcoxon(diffs, alternative="two-sided")
    return float(np.clip(p, 0.0, 1.0))


def test_wilcoxon_clear_difference_low_p():
    """If Regime-Markov distribution is much lower than baseline, p < 0.05."""
    np.random.seed(42)
    method = np.random.normal(0.10, 0.02, size=199)
    baseline = np.random.normal(0.20, 0.02, size=199)
    p = compute_wilcoxon_pvalue(method, baseline)
    assert p < 0.001


def test_wilcoxon_no_difference_high_p():
    """If distributions are identical, p > 0.5."""
    np.random.seed(42)
    # Same exact data for both - differences are all zero, test should return high p-value
    data = np.random.normal(0.10, 0.02, size=199)
    p = compute_wilcoxon_pvalue(data, data.copy())
    assert p > 0.5


def test_wilcoxon_short_array_returns_1():
    """If either array has < 2 elements, return 1.0 (no information)."""
    p = compute_wilcoxon_pvalue(np.array([0.1]), np.array([0.2, 0.3]))
    assert p == 1.0


def test_wilcoxon_constant_returns_1():
    """If both arrays are constant (no variation), return 1.0."""
    p = compute_wilcoxon_pvalue(np.array([0.1] * 100), np.array([0.1] * 100))
    assert p == 1.0