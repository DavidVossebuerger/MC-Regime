"""Paired Wilcoxon signed-rank test for structure-preservation losses."""

import numpy as np
from scipy.stats import wilcoxon


def paired_loss_test(
    loss_method_a: np.ndarray,
    loss_method_b: np.ndarray,
    alternative: str = "less",
) -> dict:
    """Wilcoxon signed-rank test on paired losses.

    H0: median(a - b) >= 0 (method a is not better)
    H1: median(a - b) < 0 (method a is better, lower loss)

    Args:
        loss_method_a: Array of losses for method A (regime bootstrap)
        loss_method_b: Array of losses for method B (baseline)
        alternative: Test direction - "less" = test a < b, "greater" = test a > b

    Returns:
        dict with keys: p_value, statistic, effect_size (Cohen's d on differences),
                       median_difference
    """
    loss_method_a = np.asarray(loss_method_a).flatten()
    loss_method_b = np.asarray(loss_method_b).flatten()

    if loss_method_a.shape != loss_method_b.shape:
        raise ValueError("loss_method_a and loss_method_b must have the same shape")

    differences = loss_method_a - loss_method_b

    # Handle case where all differences are zero
    if np.all(differences == 0):
        return {
            "p_value": 1.0,
            "statistic": 0.0,
            "effect_size": 0.0,
            "median_difference": 0.0,
        }

    # Wilcoxon signed-rank test
    # alternative='less' tests H1: median(a - b) < 0 i.e. method a has lower loss
    stat, p_value = wilcoxon(differences, alternative=alternative)

    # Effect size: Cohen's d on the differences
    effect_size = cohens_d_from_differences(differences)

    # Median difference
    median_diff = np.median(differences)

    return {
        "p_value": float(p_value),
        "statistic": float(stat),
        "effect_size": float(effect_size),
        "median_difference": float(median_diff),
    }


def cohens_d_from_differences(differences: np.ndarray) -> float:
    """Compute Cohen's d from paired differences."""
    diff_mean = np.mean(differences)
    diff_std = np.std(differences, ddof=1)

    if diff_std == 0:
        return 0.0

    # Cohen's d = mean difference / std of differences
    d = diff_mean / diff_std
    return d