"""Effect size computation (Cohen's d)."""

import numpy as np


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples.

    Cohen's d measures the standardized difference between two groups:
    d = (mean_a - mean_b) / pooled_std

    Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

    Args:
        group_a: First group (e.g., treatment/bootstrapped)
        group_b: Second group (e.g., control/baseline)

    Returns:
        Cohen's d (positive when group_a > group_b)
    """
    group_a = np.asarray(group_a).flatten()
    group_b = np.asarray(group_b).flatten()

    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("Both groups must have at least 2 elements")

    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    # Pooled standard deviation
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    n_a = len(group_a)
    n_b = len(group_b)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean_a - mean_b) / pooled_std
    return d


def cohens_d_from_summary(
    mean_a: float,
    mean_b: float,
    std_a: float,
    std_b: float,
    n_a: int,
    n_b: int,
) -> float:
    """Compute Cohen's d from summary statistics.

    Args:
        mean_a: Mean of group A
        mean_b: Mean of group B
        std_a: Standard deviation of group A
        std_b: Standard deviation of group B
        n_a: Sample size of group A
        n_b: Sample size of group B

    Returns:
        Cohen's d
    """
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean_a - mean_b) / pooled_std
    return d


def glass_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Glass's delta: standardized difference using control group SD.

    Useful when control group is the reference.

    Args:
        group_a: Treatment group
        group_b: Control group

    Returns:
        Glass's delta
    """
    group_a = np.asarray(group_a).flatten()
    group_b = np.asarray(group_b).flatten()

    std_b = np.std(group_b, ddof=1)

    if std_b == 0:
        return 0.0

    return (np.mean(group_a) - np.mean(group_b)) / std_b