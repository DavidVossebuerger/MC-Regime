"""Level 3: Network structure metrics.

This module builds correlation graphs from variable pairs and computes
distances between original and bootstrap networks using Frobenius
and spectral norms.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_correlation_graph(
    data: pd.DataFrame,
    nodes: "Sequence[str]" | None = None,
    metric: str = "absolute",
) -> np.ndarray:
    """Build an n×n correlation adjacency matrix.

    Args:
        data: DataFrame with columns for each variable.
        nodes: List of variable names to include. If None, uses all numeric columns.
        metric: 'absolute' for |corr|, 'signed' for raw corr values.

    Returns:
        Symmetric n×n numpy array where A[i,j] = correlation(v_i, v_j).
        Diagonal is always 1.0.
    """
    if nodes is None:
        # Use all numeric columns
        nodes = list(data.select_dtypes(include=[np.number]).columns)

    n = len(nodes)
    A = np.eye(n)  # Start with identity (diagonal = 1)

    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = nodes[i], nodes[j]
            if v1 in data.columns and v2 in data.columns:
                # Drop NaN for correlation computation
                valid_mask = ~(data[v1].isna() | data[v2].isna())
                if valid_mask.sum() < 2:
                    corr = 0.0
                else:
                    corr = data.loc[valid_mask, v1].corr(data.loc[valid_mask, v2])
                    if np.isnan(corr):
                        corr = 0.0
                if metric == "absolute":
                    corr = abs(corr)
                A[i, j] = corr
                A[j, i] = corr
            else:
                # Variable not found - set to 0
                A[i, j] = 0.0
                A[j, i] = 0.0

    return A


def compute_network_distance(
    A1: np.ndarray,
    A2: np.ndarray,
    metric: str = "frobenius",
) -> float:
    """Compute distance between two adjacency matrices.

    Args:
        A1: First n×n adjacency matrix.
        A2: Second n×n adjacency matrix.
        metric: 'frobenius' for Frobenius norm of difference,
                'spectral' for eigenvalue L2 distance.

    Returns:
        Distance as a float.
    """
    if A1.shape != A2.shape:
        raise ValueError("Matrices must have the same shape")

    if metric == "frobenius":
        # Frobenius norm: sqrt(sum(A_ij^2)) = sqrt(sum((A1-A2)^2))
        diff = A1 - A2
        dist = np.sqrt(np.sum(diff**2))
        return dist

    elif metric == "spectral":
        # Eigenvalue-based distance: L2 norm of eigenvalue differences
        # First compute eigenvalues (need to ensure matrix is valid/symmetric)
        # Replace NaN with 0 and ensure symmetry
        A1 = np.nan_to_num(A1, nan=0.0)
        A2 = np.nan_to_num(A2, nan=0.0)

        # Make sure symmetric
        A1 = (A1 + A1.T) / 2
        A2 = (A2 + A2.T) / 2

        try:
            eig1 = np.linalg.eigvalsh(A1)
            eig2 = np.linalg.eigvalsh(A2)
        except np.linalg.LinAlgError:
            # Fallback: use Frobenius as proxy if eigenvalues don't converge
            diff = A1 - A2
            return np.sqrt(np.sum(diff**2))

        # Sort by absolute value descending
        eig1 = np.sort(np.abs(eig1))[::-1]
        eig2 = np.sort(np.abs(eig2))[::-1]

        # Spectral distance = L2 norm of eigenvalue differences
        dist = np.sqrt(np.sum((eig1 - eig2)**2))
        return dist

    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'frobenius' or 'spectral'.")


def compute_l3_network(
    original_graph: np.ndarray,
    bootstrap_graphs: list[np.ndarray],
    metrics: tuple[str, ...] = ("frobenius", "spectral"),
) -> dict[str, float]:
    """Compute Level 3 network metrics.

    For each graph in bootstrap_graphs, computes distance to original,
    then averages across all bootstrap replicates.

    Args:
        original_graph: n×n adjacency matrix from original data.
        bootstrap_graphs: List of n×n adjacency matrices from bootstrap.
        metrics: Tuple of metrics to compute ('frobenius', 'spectral').

    Returns:
        Dictionary with keys 'frobenius' and/or 'spectral', mapped to mean distances.
    """
    if not bootstrap_graphs:
        raise ValueError("bootstrap_graphs cannot be empty")

    n_metrics = len(metrics)
    results = {m: [] for m in metrics}

    for boot_graph in bootstrap_graphs:
        for metric in metrics:
            dist = compute_network_distance(original_graph, boot_graph, metric=metric)
            results[metric].append(dist)

    # Average across all replicates
    return {m: np.mean(results[m]) for m in metrics}