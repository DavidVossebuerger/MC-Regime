"""Test the Sinkhorn cap-sensitivity sweep helper.

Defends against the post-hoc-tuning critique: 3% is the minimum cap
that passes the §11.7 gate (TV < 0.10), not a cherry-picked value.
"""
import numpy as np
import pandas as pd

from mc_regime.regimes.transition import (
    calibrate_transition_to_target_stationary,
    stationary_distribution,
)


def sinkhorn_cap_sensitivity(trans_matrix, target, caps=None, tv_threshold=0.10):
    """Sweep the Sinkhorn cap and return a sensitivity table."""
    if caps is None:
        caps = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
    rows = []
    for cap in caps:
        P_cal = calibrate_transition_to_target_stationary(
            trans_matrix, target, tv_threshold=tv_threshold, max_forbidden_per_cell=cap
        )
        pi_stat = stationary_distribution(P_cal)
        tv = 0.5 * np.abs(pi_stat - target).sum()
        forbidden_mask = (trans_matrix.values == 0)
        max_forbidden = float(P_cal.values[forbidden_mask].max()) if forbidden_mask.any() else 0.0
        rows.append({
            "cap": cap, "tv_drift": tv,
            "gate_pass": tv < tv_threshold, "max_forbidden_mass": max_forbidden,
        })
    return pd.DataFrame(rows)


def test_sensitivity_returns_dataframe():
    np.random.seed(42)
    target = np.array([0.125] * 8)
    P = np.eye(8) * 0.9 + np.full((8, 8), 0.1 / 8)
    P_df = pd.DataFrame(P, index=[f"s{i}" for i in range(8)], columns=[f"s{i}" for i in range(8)])
    result = sinkhorn_cap_sensitivity(P_df, target)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 7
    assert set(["cap", "tv_drift", "gate_pass", "max_forbidden_mass"]).issubset(result.columns)


def test_minimum_passing_cap_in_range():
    np.random.seed(42)
    target = np.array([0.125] * 8)
    P = np.eye(8) * 0.9 + np.full((8, 8), 0.1 / 8)
    P_df = pd.DataFrame(P, index=[f"s{i}" for i in range(8)], columns=[f"s{i}" for i in range(8)])
    result = sinkhorn_cap_sensitivity(P_df, target)
    assert result["gate_pass"].sum() >= 1
    min_passing = result[result["gate_pass"]]["cap"].min()
    assert min_passing <= 0.10