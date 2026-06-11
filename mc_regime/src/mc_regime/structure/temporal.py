"""Temporal / path-dependent structure metrics.

Per Spec §3 + BUG-3-fix: IID trivially preserves *contemporaneous*
correlations (L1, L2, L3_fro), so Markov-ordering can never beat it on
those levels. The right tests are *temporal* — does the simulated
sequence preserve autocorrelation, lagged cross-correlations, drawdown
distributions? These are the metrics where Markov chain structure should
outperform IID.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation of a 1-D series."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return 0.0
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()
    denom = np.sqrt((x0 ** 2).sum() * (x1 ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((x0 * x1).sum() / denom)


def lagk_cross_corr(x: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Cross-correlation between x[t] and y[t+k] (positive k = y leads).

    Truncates both series to the joint length so x and y can have
    different lengths (e.g. y is a differenced version of x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= k or len(y) <= k or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    # Both arrays must have equal length after truncation
    n = min(len(x) - k, len(y) - k)
    if n < 2:
        return 0.0
    a = x[:n] - x[:n].mean()
    b = y[k:k + n] - y[k:k + n].mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def variance_ratio(x: np.ndarray, k: int = 2) -> float:
    """Variance ratio: Var(k-period return) / (k * Var(1-period return)).

    Under random walk, VR ≈ 1. VR < 1 = mean reversion. VR > 1 = momentum.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < k + 1:
        return 1.0
    r1 = x[1:] - x[:-1]
    rk = x[k:] - x[:-k]
    v1 = r1.var(ddof=1)
    vk = rk.var(ddof=1)
    if v1 < 1e-12:
        return 1.0
    return float(vk / (k * v1))


def max_drawdown(x: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a price/return path."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    cum = np.cumsum(x) if not np.all(x > 0) else np.log(np.cumprod(1 + x))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak)
    return float(-dd.min())


def temporal_metrics(
    fx: np.ndarray,
    fvg: np.ndarray,
) -> Dict[str, float]:
    """Bundle of temporal / path-dependent metrics on FX + FVG series.

    Args:
        fx: FX log-returns or log-levels.
        fvg: Fair-Value-Gap series.

    Returns:
        Dict with keys: fx_acf1, fx_var_ratio2, fvg_acf1, fvg_lead_FX_1,
        fvg_max_dd, fx_max_dd.
    """
    fx_arr = np.asarray(fx, dtype=float)
    fvg_arr = np.asarray(fvg, dtype=float)
    fx_ret = np.diff(fx_arr) if len(fx_arr) > 1 else np.array([0.0])
    # For lead-lag cross-corr, truncate both series to the joint length
    T = min(len(fvg_arr), len(fx_ret) + 1)
    fvg_aligned = fvg_arr[:T]
    fx_ret_aligned = fx_ret[:T - 1]
    return {
        "fx_acf1": lag1_autocorr(fx_ret),
        "fx_var_ratio2": variance_ratio(fx_ret, k=2),
        "fvg_acf1": lag1_autocorr(fvg_arr),
        "fvg_lead_FX_1": lagk_cross_corr(fvg_aligned, fx_ret_aligned, k=1),
        "fvg_max_dd": max_drawdown(fvg_arr),
        "fx_max_dd": max_drawdown(fx_ret),
    }


def temporal_distance(
    sim_fx: np.ndarray,
    sim_fvg: np.ndarray,
    orig_fx: np.ndarray,
    orig_fvg: np.ndarray,
) -> Dict[str, float]:
    """Per-replication absolute deviation of temporal metrics from original.

    Returns dict[metric_name] -> |sim - orig|. Lower = better.
    """
    sim_m = temporal_metrics(sim_fx, sim_fvg)
    orig_m = temporal_metrics(orig_fx, orig_fvg)
    return {k: abs(sim_m[k] - orig_m[k]) for k in sim_m}


def aggregate_temporal_loss(
    sim_fx_list: List[np.ndarray],
    sim_fvg_list: List[np.ndarray],
    orig_fx: np.ndarray,
    orig_fvg: np.ndarray,
) -> Dict[str, float]:
    """Mean temporal distance across many simulations."""
    return mean_temporal_loss(sim_fx_list, sim_fvg_list, orig_fx, orig_fvg)


def mean_temporal_loss(
    sim_sequences_fx: List[np.ndarray],
    sim_sequences_fvg: List[np.ndarray],
    orig_fx: np.ndarray,
    orig_fvg: np.ndarray,
) -> Dict[str, float]:
    """Mean temporal distance across many simulations."""
    if not sim_sequences_fx:
        return {}
    distances: Dict[str, List[float]] = {}
    for sim_fx, sim_fvg in zip(sim_sequences_fx, sim_sequences_fvg):
        d = temporal_distance(sim_fx, sim_fvg, orig_fx, orig_fvg)
        for k, v in d.items():
            distances.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in distances.items()}
