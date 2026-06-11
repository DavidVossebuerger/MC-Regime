"""Hold-out validation for the v14 regime-preserving bootstrap.

In v13, the bootstrap was trained and evaluated on the same
2003-2025 sample (no held-out test). A reviewer can rightly ask
whether the calibrated Markov matrix predicts regime frequencies
out-of-sample. This module provides the train/test split helpers
and the regime-frequency comparison function.

Usage in main pipeline:
    train_dates, test_dates = split_train_test_dates(all_dates, cutoff=datetime(2024, 1, 1))
    result = compute_hold_out_regime_frequencies(states_train, states_test, n_states=8)
"""
from datetime import datetime
from typing import Tuple
import pandas as pd
import numpy as np


def split_train_test_dates(
    dates: pd.DatetimeIndex,
    cutoff: datetime = datetime(2024, 1, 1),
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split dates into train (before cutoff) and test (on/after cutoff).

    Args:
        dates: full date range.
        cutoff: dates strictly before this are train; on/after are test.

    Returns:
        (train_dates, test_dates).
    """
    train = dates[dates < cutoff]
    test = dates[dates >= cutoff]
    return train, test


def compute_hold_out_regime_frequencies(
    states_train: np.ndarray,
    states_test: np.ndarray,
    n_states: int = 8,
) -> dict:
    """Compare regime frequencies in train and test sets.

    Returns dict with: train_freq (length n_states), test_freq (length
    n_states), tv_drift (total variation between them).
    """
    states_train = np.asarray(states_train).flatten()
    states_test = np.asarray(states_test).flatten()
    train_freq = np.bincount(states_train.astype(int), minlength=n_states) / max(len(states_train), 1)
    test_freq = np.bincount(states_test.astype(int), minlength=n_states) / max(len(states_test), 1)
    tv = 0.5 * np.abs(train_freq - test_freq).sum()
    return {
        "train_freq": train_freq.tolist(),
        "test_freq": test_freq.tolist(),
        "tv_drift": float(tv),
    }