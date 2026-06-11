"""Unit tests for HMM block construction."""

import numpy as np
import pandas as pd
import pytest

from mc_regime.regimes.base import Block
from mc_regime.regimes.hmm_blocks import HMMBlockProvider


class TestHMMBlockProvider:
    """Test suite for HMMBlockProvider."""

    def test_single_block(self):
        """Test with all same states produces single block."""
        states = pd.Series([0] * 10, index=pd.date_range("2020-01-01", periods=10, freq="ME"))
        provider = HMMBlockProvider(states)

        blocks = provider.get_blocks()
        assert len(blocks) == 1
        assert blocks[0].regime_id == 0
        assert blocks[0].start == pd.Timestamp("2020-01-31")
        assert blocks[0].end == pd.Timestamp("2020-10-31")

    def test_two_blocks(self):
        """Test with a single regime change produces two blocks."""
        states_arr = [0] * 5 + [1] * 5
        states = pd.Series(states_arr, index=pd.date_range("2020-01-01", periods=10, freq="ME"))
        provider = HMMBlockProvider(states)

        blocks = provider.get_blocks()
        assert len(blocks) == 2
        assert blocks[0].regime_id == 0
        assert blocks[1].regime_id == 1
        # Check contiguity
        assert blocks[0].end < blocks[1].start

    def test_multiple_regime_changes(self):
        """Test with multiple regime changes."""
        states_arr = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        dates = pd.date_range("2020-01-01", periods=10, freq="ME")
        states = pd.Series(states_arr, index=dates)
        provider = HMMBlockProvider(states)

        blocks = provider.get_blocks()
        assert len(blocks) == 5  # 5 blocks total

    def test_block_properties(self):
        """Test block properties."""
        states = pd.Series([0] * 5 + [1] * 5, index=pd.date_range("2020-01-01", periods=10, freq="ME"))
        provider = HMMBlockProvider(states)

        blocks = provider.get_blocks()
        # First block: Jan-May (5 months)
        assert blocks[0].length_months == 5
        # Second block: Jun-Oct (5 months)
        assert blocks[1].length_months == 5

    def test_n_states(self):
        """Test n_states property."""
        states_arr = [0, 0, 1, 1, 1, 2, 2, 2]
        states = pd.Series(states_arr, index=pd.date_range("2020-01-01", periods=8, freq="ME"))
        provider = HMMBlockProvider(states)

        assert provider.n_states == 3

    def test_empty_states(self):
        """Test with empty states series."""
        states = pd.Series([], dtype=int)
        provider = HMMBlockProvider(states)

        blocks = provider.get_blocks()
        assert len(blocks) == 0

    def test_date_filtering(self):
        """Test date range filtering."""
        states_arr = [0] * 10 + [1] * 10
        dates = pd.date_range("2010-01-01", periods=20, freq="ME")
        states = pd.Series(states_arr, index=dates)
        provider = HMMBlockProvider(states)

        # Filter to first 12 months
        filtered = provider.get_blocks(
            start_date=pd.Timestamp("2010-01-01"),
            end_date=pd.Timestamp("2010-12-31")
        )

        # Should get first block spanning 2010-01 to 2010-10
        assert len(filtered) >= 1


class TestTransitionMatrix:
    """Test transition matrix estimation."""

    def test_laplace_smoothing(self):
        """Test that Laplace smoothing adds small probability to all entries."""
        # Perfect persistence: all 0s except last one
        states = [0] * 9 + [1]
        states = pd.Series(states, index=pd.date_range("2020-01-01", periods=10, freq="ME"))
        provider = HMMBlockProvider(states)

        tm = provider.transition_matrix

        # With Laplace smoothing, all entries should be > 0
        assert (tm.values > 0).all()

    def test_rows_sum_to_one(self):
        """Test that transition matrix rows sum to 1."""
        states_arr = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
        states = pd.Series(states_arr, index=pd.date_range("2020-01-01", periods=10, freq="ME"))
        provider = HMMBlockProvider(states)

        tm = provider.transition_matrix
        row_sums = tm.sum(axis=1).values

        assert np.allclose(row_sums, 1.0)

    def test_diagonal_greater_than_off_diagonal(self):
        """Test that diagonal > off-diagonal for persistent regimes."""
        # Highly persistent sequence
        states = [0] * 20 + [1] * 20 + [0] * 20
        states = pd.Series(states, index=pd.date_range("2010-01-01", periods=60, freq="ME"))
        provider = HMMBlockProvider(states)

        tm = provider.transition_matrix

        # Diagonal should dominate
        for i in range(provider.n_states):
            for j in range(provider.n_states):
                if i != j:
                    assert tm.iloc[i, i] > tm.iloc[i, j]


class TestBlocksEdgeCases:
    """Edge case tests."""

    def test_regime_oscillation(self):
        """Test with oscillating states (alternating)."""
        states_arr = [0, 1, 0, 1, 0, 1, 0, 1]
        dates = pd.date_range("2020-01-01", periods=8, freq="ME")
        states = pd.Series(states_arr, index=dates)
        provider = HMMBlockProvider(states)

        blocks = provider.get_blocks()
        # Should get 8 blocks when oscillating
        assert len(blocks) == 8

    def test_single_observation(self):
        """Test with single observation."""
        states = pd.Series([0], index=[pd.Timestamp("2020-01-31")])
        provider = HMMBlockProvider(states)

        blocks = provider.get_blocks()
        assert len(blocks) == 1
        assert blocks[0].length_months == 1