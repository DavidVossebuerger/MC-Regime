"""Arrangement-distance metrics between regime sequences.

Per Spec §5 + §6: the **arrangement** of a simulated regime sequence must
be measurably distant from the original. Three complementary metrics:

- composition_distance : how often each regime appears (histogram L1)
- transition_distance  : how often each transition appears (matrix L1)
- edit_distance        : Levenshtein distance of label sequences

All return non-negative scalars; larger = more distant.
"""

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _labels_to_sequence(labels: Sequence[int]) -> List[int]:
    return [int(x) for x in labels]


def composition_distance(
    sim_labels: Sequence[int],
    orig_labels: Sequence[int],
    n_regimes: Optional[int] = None,
) -> float:
    """L1 distance of regime-frequency histograms.

    Args:
        sim_labels: Regime IDs in the simulated path.
        orig_labels: Regime IDs in the original path.
        n_regimes: Number of distinct regimes (defaults to max+1 across both).

    Returns:
        Sum_k |freq_sim[k] - freq_orig[k]| in [0, 2].
    """
    if n_regimes is None:
        n_regimes = max(max(sim_labels, default=0), max(orig_labels, default=0)) + 1
    sim_counts = np.bincount(np.asarray(sim_labels, dtype=int), minlength=n_regimes)
    orig_counts = np.bincount(np.asarray(orig_labels, dtype=int), minlength=n_regimes)
    sim_freq = sim_counts / max(sim_counts.sum(), 1)
    orig_freq = orig_counts / max(orig_counts.sum(), 1)
    return float(np.sum(np.abs(sim_freq - orig_freq)))


def transition_distance(
    sim_labels: Sequence[int],
    orig_labels: Sequence[int],
    n_regimes: Optional[int] = None,
) -> float:
    """L1 distance of transition-count matrices.

    Normalised to [0, 2]: 0 = identical transition structure, 2 = no overlap.
    """
    if n_regimes is None:
        n_regimes = max(max(sim_labels, default=0), max(orig_labels, default=0)) + 1

    def _trans_matrix(labels):
        M = np.zeros((n_regimes, n_regimes))
        for t in range(len(labels) - 1):
            i, j = int(labels[t]), int(labels[t + 1])
            if i < n_regimes and j < n_regimes:
                M[i, j] += 1
        # Normalise rows
        for i in range(n_regimes):
            s = M[i].sum()
            if s > 0:
                M[i] /= s
        return M

    sim_T = _trans_matrix(sim_labels)
    orig_T = _trans_matrix(orig_labels)
    return float(np.sum(np.abs(sim_T - orig_T)))


def edit_distance(
    sim_labels: Sequence[int],
    orig_labels: Sequence[int],
) -> int:
    """Levenshtein (edit) distance between two label sequences.

    Returns the minimum number of single-element insertions, deletions,
    or substitutions to turn sim_labels into orig_labels.
    """
    s = _labels_to_sequence(sim_labels)
    o = _labels_to_sequence(orig_labels)
    n, m = len(s), len(o)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == o[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,        # deletion
                dp[i, j - 1] + 1,        # insertion
                dp[i - 1, j - 1] + cost,  # substitution
            )
    return int(dp[n, m])


def normalised_edit_distance(
    sim_labels: Sequence[int],
    orig_labels: Sequence[int],
) -> float:
    """Edit distance normalised by max(len(sim), len(orig)) in [0, 1]."""
    ed = edit_distance(sim_labels, orig_labels)
    n = max(len(sim_labels), len(orig_labels))
    return ed / max(n, 1)


def arrangement_distance(
    sim_labels: Sequence[int],
    orig_labels: Sequence[int],
    metric: str = "transition",
    n_regimes: Optional[int] = None,
) -> float:
    """Compute arrangement distance between two regime sequences.

    Args:
        sim_labels: Regime IDs in simulated path.
        orig_labels: Regime IDs in original path.
        metric: One of "composition", "transition", "edit", "norm_edit".
        n_regimes: Number of regimes.

    Returns:
        Non-negative distance (interpretation depends on metric).
    """
    if metric == "composition":
        return composition_distance(sim_labels, orig_labels, n_regimes)
    if metric == "transition":
        return transition_distance(sim_labels, orig_labels, n_regimes)
    if metric == "edit":
        return float(edit_distance(sim_labels, orig_labels))
    if metric == "norm_edit":
        return normalised_edit_distance(sim_labels, orig_labels)
    raise ValueError(f"Unknown metric: {metric}")


def compute_arrangement_distances(
    sim_sequences: Sequence[Sequence[int]],
    orig_labels: Sequence[int],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute arrangement distances for many simulated sequences.

    Returns:
        DataFrame with one row per simulation and one column per metric.
    """
    if metrics is None:
        metrics = ["composition", "transition", "norm_edit"]
    rows = []
    for sim in sim_sequences:
        row = {}
        for m in metrics:
            row[m] = arrangement_distance(sim, orig_labels, metric=m)
        rows.append(row)
    return pd.DataFrame(rows)
