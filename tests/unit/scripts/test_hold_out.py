"""Test the hold-out split logic (v14 Reviewer §4 fix)."""
import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from mc_regime.scripts.hold_out import (
    split_train_test_dates,
    compute_hold_out_regime_frequencies,
)


def test_train_test_split_2024_cutoff():
    """Default cutoff: train <= 2023-12-31, test >= 2024-01-01."""
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    train, test = split_train_test_dates(dates, cutoff=datetime(2024, 1, 1))
    assert train.max() < datetime(2024, 1, 1)
    assert test.min() >= datetime(2024, 1, 1)
    assert len(train) + len(test) == len(dates)


def test_custom_cutoff():
    """Custom cutoff is respected."""
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    train, test = split_train_test_dates(dates, cutoff=datetime(2022, 6, 15))
    assert train.max() < datetime(2022, 6, 15)
    assert test.min() >= datetime(2022, 6, 15)


def test_regime_frequency_computation():
    """compute_hold_out_regime_frequencies returns train_freq, test_freq, tv_drift."""
    states_train = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 2])
    states_test = np.array([0, 1, 2, 0, 1])
    result = compute_hold_out_regime_frequencies(states_train, states_test, n_states=3)
    assert "train_freq" in result
    assert "test_freq" in result
    assert "tv_drift" in result
    assert len(result["train_freq"]) == 3
    assert len(result["test_freq"]) == 3
    assert 0.0 <= result["tv_drift"] <= 1.0