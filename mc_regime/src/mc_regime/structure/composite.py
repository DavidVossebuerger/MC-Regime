"""Composite structure preservation metrics.

This module computes the composite loss as a weighted average of Level 1-3 metrics.
Primary report is individual levels; composite is supplementary.
"""
from __future__ import annotations

import numpy as np


def compute_composite_loss(
    L1: float,
    L2: float,
    L3_fro: float,
    L3_spec: float,
    weights: tuple[float, float, float, float] | None = None,
) -> float:
    """Compute composite loss as weighted average of structure metrics.

    Composite L = (w1*L1 + w2*L2 + w3_fro*L3_fro + w3_spec*L3_spec) / sum(weights)

    Args:
        L1: Level 1 correlation bias.
        L2: Level 2 Wasserstein distance.
        L3_fro: Level 3 Frobenius network distance.
        L3_spec: Level 3 spectral network distance.
        weights: Tuple of (w1, w2, w3_fro, w3_spec).
                Defaults to (0.25, 0.25, 0.25, 0.25).

    Returns:
        Composite loss as float.
    """
    if weights is None:
        weights = (0.25, 0.25, 0.25, 0.25)

    if len(weights) != 4:
        raise ValueError("weights must have 4 elements")

    w_total = sum(weights)
    composite = (
        weights[0] * L1 +
        weights[1] * L2 +
        weights[2] * L3_fro +
        weights[3] * L3_spec
    ) / w_total

    return composite


def structure_preservation_report(
    L1: float,
    L2: float,
    L3_fro: float,
    L3_spec: float,
    composite: float | None = None,
) -> dict:
    """Generate a formatted report dictionary.

    Args:
        L1: Level 1 correlation bias.
        L2: Level 2 Wasserstein distance.
        L3_fro: Level 3 Frobenius network distance.
        L3_spec: Level 3 spectral network distance.
        composite: Optional composite loss. If None, computes default.

    Returns:
        Dictionary with all metrics.
    """
    if composite is None:
        composite = compute_composite_loss(L1, L2, L3_fro, L3_spec)

    return {
        "L1_correlation_bias": L1,
        "L2_wasserstein": L2,
        "L3_frobenius": L3_fro,
        "L3_spectral": L3_spec,
        "composite": composite,
    }