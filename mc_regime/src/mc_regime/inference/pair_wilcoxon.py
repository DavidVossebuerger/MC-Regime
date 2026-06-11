"""Pair-level statistical inference for v14.

v13 used scipy.stats.wilcoxon over (3 pairs × 1999 replications) = 5997
samples. This was pseudo-replication: the 1999 replications are NOT
independent draws from the population.

v14 fix: aggregate losses to ONE number per (method, metric, pair),
then run a paired sign test (3 paired observations per test). This is
the correct inference: 3 independent FX pairs, 3 paired comparisons.
"""
import numpy as np
from scipy.stats import wilcoxon


def pair_level_pvalue(method_per_pair: np.ndarray, baseline_per_pair: np.ndarray) -> float:
    """Paired sign test (Wilcoxon signed-rank) of n=3 paired observations."""
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


def pair_level_effect_size(method_per_pair: np.ndarray, baseline_per_pair: np.ndarray) -> dict:
    """Report effect size and bootstrap CI for (method - baseline) difference."""
    method_per_pair = np.asarray(method_per_pair, dtype=float).flatten()
    baseline_per_pair = np.asarray(baseline_per_pair, dtype=float).flatten()
    diffs = method_per_pair - baseline_per_pair
    n = len(diffs)
    np.random.seed(42)
    n_boot = 10000
    boot_means = np.array([
        np.mean(np.random.choice(diffs, size=n, replace=True))
        for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_pairs": int(n),
    }