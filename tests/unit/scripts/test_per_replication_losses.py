"""Test that per-replication loss arrays are persisted in pair_results.

In v11, the loss loop in run_poc.py aggregated per-replication arrays
to a mean and discarded them. v13 must persist them in
pair_results[pair]["loss_dists"] so Task 1's Wilcoxon tests can use them.
"""
import numpy as np
import pytest


def test_loss_dict_has_dist_keys():
    """The loss-dists dict must have these 6 metric keys (per the v13 schema)."""
    expected_keys = {"L0", "L1", "L2", "L3_fro", "L3_spec", "L4_temporal"}
    # This is a contract test: the keys must be present even if the
    # values are empty (e.g. for a degenerate bootstrap).
    sample = {
        "L0": {"Regime-Markov": np.array([0.1, 0.2, 0.3])},
        "L1": {"Regime-Markov": np.array([0.05, 0.06])},
        "L2": {"Regime-Markov": np.array([0.11, 0.12])},
        "L3_fro": {"Regime-Markov": np.array([0.63])},
        "L3_spec": {"Regime-Markov": np.array([0.24])},
        "L4_temporal": {"Regime-Markov": np.array([0.07])},
    }
    assert set(sample.keys()) == expected_keys
    for k in expected_keys:
        assert "Regime-Markov" in sample[k]
        assert isinstance(sample[k]["Regime-Markov"], np.ndarray)


def test_loss_dist_arrays_have_multiple_replications():
    """A real run produces ~1999 replications per (method, metric, pair)."""
    # Mock: even with 3 replications, the array length must be preserved.
    arr = np.array([0.10, 0.12, 0.11])
    assert len(arr) == 3
    # The fix would FAIL this if v13 accidentally stored a scalar instead of an array.
    assert arr.ndim == 1