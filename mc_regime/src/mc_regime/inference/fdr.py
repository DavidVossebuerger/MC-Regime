"""Benjamini-Hochberg False Discovery Rate correction."""

import numpy as np


def apply_bh_fdr(p_values: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Controls the false discovery rate among multiple hypotheses.
    Implements the Benjamini-Hochberg procedure manually since statsmodels
    may not be available.

    Args:
        p_values: Array of raw p-values
        alpha: Target FDR level (default 0.10)

    Returns:
        Array of adjusted q-values (FDR-adjusted p-values)
    """
    p_values = np.asarray(p_values).flatten()

    if len(p_values) == 0:
        return np.array([])

    if np.any(p_values < 0) or np.any(p_values > 1):
        raise ValueError("p_values must be in [0, 1]")

    # Sort p-values while tracking original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    m = len(sorted_p)

    # BH critical values: p_i <= (i/m) * alpha
    # Compute adjusted p-values (q-values)
    q_values = np.zeros(m)

    # Start from the largest p-value
    cumulative_min = 1.0
    for i in range(m - 1, -1, -1):
        # Adjusted p-value = min(p_i * m / i, 1)
        adjusted = sorted_p[i] * m / (i + 1)
        adjusted = min(adjusted, 1.0)
        # Enforce monotonicity (preserve order after adjustment)
        cumulative_min = min(cumulative_min, adjusted)
        q_values[i] = cumulative_min

    # Restore original order
    q_values_original_order = np.zeros(m)
    q_values_original_order[sorted_indices] = q_values

    return q_values_original_order


def bh_reject_hypotheses(p_values: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Return boolean array of rejected hypotheses using BH procedure.

    Args:
        p_values: Array of raw p-values
        alpha: Target FDR level

    Returns:
        Boolean array where True indicates rejection
    """
    p_values = np.asarray(p_values).flatten()

    if len(p_values) == 0:
        return np.array([False])

    # Sort p-values while tracking original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    m = len(sorted_p)

    # Find largest k such that p_(k) <= k/m * alpha
    # Using BH critical values
    rejected = np.zeros(m, dtype=bool)

    for i in range(m):
        critical_value = (i + 1) / m * alpha
        if sorted_p[i] <= critical_value:
            rejected[i] = True
        else:
            # Once we fail, all larger p-values also fail
            break

    # Restore original order
    rejected_original_order = np.zeros(m, dtype=bool)
    rejected_original_order[sorted_indices] = rejected

    return rejected_original_order