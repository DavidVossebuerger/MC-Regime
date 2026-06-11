"""BCa (Bias-Corrected and Accelerated) bootstrap p-value computation."""

import warnings

warnings.warn(
    "mc_regime.inference.bca is deprecated as of v13; the BCa jackknife "
    "acceleration factor is statistically biased with <100 blocks (we use 24). "
    "v13 uses scipy.stats.wilcoxon over per-replication loss arrays. "
    "Kept for backwards compat; do not import in new code.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from scipy.stats import norm


def bca_p_value(
    observed: float,
    bootstrap_stat: np.ndarray,
    jackknife_stat: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """BCa p-value with jackknife-based acceleration.

    Implements the Bias-Corrected and Accelerated (BCa) bootstrap method
    for accurate p-value estimation.

    Args:
        observed: The observed test statistic value
        bootstrap_stat: Array of bootstrap replicate statistics
        jackknife_stat: Array of jackknife leave-one-out statistics
        alpha: Significance level for two-sided test (default 0.05)

    Returns:
        BCa-corrected p-value in [0, 1]
    """
    bootstrap_stat = np.asarray(bootstrap_stat).flatten()
    jackknife_stat = np.asarray(jackknife_stat).flatten()

    n = len(bootstrap_stat)
    m = len(jackknife_stat)

    if n < 2 or m < 2:
        raise ValueError("bootstrap_stat and jackknife_stat must have at least 2 elements")

    # Step 1: Bias correction factor (z0)
    # Proportion of bootstrap values less than the observed
    below_count = np.sum(bootstrap_stat < observed)
    below_prop = below_count / n
    z0 = norm.ppf(below_prop)

    # Step 2: Acceleration factor (a)
    # Based on jackknife estimate of skewness
    theta_dot = np.mean(jackknife_stat)
    num = np.sum((theta_dot - jackknife_stat) ** 3)
    denom = 6 * (np.sum((theta_dot - jackknife_stat) ** 2) ** 1.5)

    if denom == 0:
        a = 0.0
    else:
        a = num / denom

    # Clamp acceleration to avoid numerical instability
    a = np.clip(a, -0.25, 0.25)

    # Step 3: Compute BCa p-value using percentile method with correction
    # Alternative approach: use bias-corrected percentile
    # Adjusted percentile for observed value
    try:
        # Lower tail: proportion of bootstrap values <= observed
        p_lower = norm.cdf(z0 + (z0 + norm.ppf(alpha / 2)) / (1 - a * (z0 + norm.ppf(alpha / 2))))
        p_upper = norm.cdf(z0 + (z0 + norm.ppf(1 - alpha / 2)) / (1 - a * (z0 + norm.ppf(1 - alpha / 2))))

        # If calculation fails, fall back to simple percentile
        if not np.isfinite(p_lower) or not np.isfinite(p_upper):
            # Fall back to simple bias-corrected percentile
            p_lower = below_prop
            p_upper = below_prop
    except (ValueError, ZeroDivisionError):
        # Fall back to simple bias-corrected percentile
        p_lower = below_prop
        p_upper = below_prop

    # Two-sided p-value
    p_value = 2 * min(p_lower, 1 - p_lower)

    return float(np.clip(p_value, 0.0, 1.0))


def bca_percentile_interval(
    bootstrap_stat: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute BCa bias-corrected percentile interval.

    Args:
        bootstrap_stat: Array of bootstrap replicates
        alpha: Significance level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_stat = np.asarray(bootstrap_stat).flatten()

    # Simple bias-corrected percentiles (BCa without acceleration)
    z0 = norm.ppf(np.mean(bootstrap_stat < np.median(bootstrap_stat)))

    alpha_lower = alpha / 2
    alpha_upper = 1 - alpha / 2

    # Adjusted percentiles
    adj_lower = norm.cdf(z0 + norm.ppf(alpha_lower))
    adj_upper = norm.cdf(z0 + norm.ppf(alpha_upper))

    lower = np.percentile(bootstrap_stat, adj_lower * 100)
    upper = np.percentile(bootstrap_stat, adj_upper * 100)

    return float(lower), float(upper)