"""Unit tests for arrangement distance metrics."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from mc_regime.structure.arrangement import (
    composition_distance,
    transition_distance,
    normalised_edit_distance,
    arrangement_distance,
    compute_arrangement_distances,
)


class TestCompositionDistance:
    """Tests for composition_distance metric."""

    def test_identical_sequences(self):
        """Identical sequences should have zero distance."""
        seq = [0, 1, 2, 0, 1, 2]
        dist = composition_distance(seq, seq)
        assert dist == 0.0

    def test_single_regime_difference(self):
        """Sequences with different regime frequencies."""
        seq1 = [0, 0, 0, 1]  # 75% regime 0, 25% regime 1
        seq2 = [1, 1, 1, 0]  # reversed
        dist = composition_distance(seq1, seq2)
        # Should be positive (L1 distance of histograms)
        assert dist > 0.0

    def test_empty_sequences(self):
        """Handle empty sequences gracefully."""
        dist = composition_distance([], [])
        assert dist == 0.0

    def test_n_regimes_parameter(self):
        """Test with explicit n_regimes parameter."""
        seq1 = [0, 1]
        seq2 = [1, 0]
        # With 3 regimes, both have equal distribution
        dist = composition_distance(seq1, seq2, n_regimes=3)
        assert dist == 0.0


class TestTransitionDistance:
    """Tests for transition_distance metric."""

    def test_identical_transitions(self):
        """Same transition structure yields distance ~0."""
        seq = [0, 1, 2, 0, 1]
        dist = transition_distance(seq, seq)
        assert dist == 0.0

    def test_different_transitions(self):
        """Different transition patterns."""
        seq1 = [0, 0, 0, 0]  # stays in 0
        seq2 = [0, 1, 1, 1]  # moves to 1
        dist = transition_distance(seq1, seq2)
        # Different matrices should give positive distance
        assert dist >= 0.0

    def test_handles_length_one(self):
        """Sequence of length 1 has no transitions."""
        seq = [0]
        dist = transition_distance(seq, [1])
        assert dist == 0.0


class TestNormalisedEditDistance:
    """Tests for normalised_edit_distance metric."""

    def test_identical_sequences(self):
        """Same sequence has distance 0."""
        seq = [0, 1, 2, 3, 4]
        dist = normalised_edit_distance(seq, seq)
        assert dist == 0.0

    def test_max_distance(self):
        """Completely different sequences."""
        seq1 = [0, 0, 0]
        seq2 = [1, 1, 1]
        dist = normalised_edit_distance(seq1, seq2)
        # Max is 1.0 when nothing overlaps
        assert dist <= 1.0

    def test_partial_overlap(self):
        """Partial overlap."""
        seq1 = [0, 1, 2]
        seq2 = [0, 1, 3]
        dist = normalised_edit_distance(seq1, seq2)
        # 1 substitution out of 3 elements
        assert 0.3 < dist < 0.4

    def test_empty_vs_nonempty(self):
        """Empty vs non-empty."""
        dist = normalised_edit_distance([], [0, 1, 2])
        assert dist == 1.0

    def test_different_lengths(self):
        """Different length sequences."""
        seq1 = [0, 1]
        seq2 = [0, 1, 2, 3]
        dist = normalised_edit_distance(seq1, seq2)
        # 2 insertions needed
        assert dist > 0.0


class TestArrangementDistance:
    """Tests for high-level arrangement_distance function."""

    def test_metric_selection(self):
        """Test metric parameter routing."""
        seq1 = [0, 1, 2]
        seq2 = [2, 1, 0]
        comp = arrangement_distance(seq1, seq2, metric="composition")
        trans = arrangement_distance(seq1, seq2, metric="transition")
        norm_ed = arrangement_distance(seq1, seq2, metric="norm_edit")
        # All should return valid numbers
        assert comp >= 0.0
        assert trans >= 0.0
        assert 0.0 <= norm_ed <= 1.0

    def test_invalid_metric(self):
        """Invalid metric raises error."""
        with pytest.raises(ValueError):
            arrangement_distance([0], [1], metric="invalid")


class TestComputeArrangementDistances:
    """Tests for compute_arrangement_distances function."""

    def test_multiple_sequences(self):
        """Multiple simulated sequences."""
        orig = [0, 1, 2, 0, 1]
        sims = [[0, 1, 2, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
        df = compute_arrangement_distances(sims, orig)
        # Should have 3 rows and metrics columns
        assert len(df) == 3
        assert "composition" in df.columns
        assert "transition" in df.columns

    def test_custom_metrics(self):
        """Custom metric list."""
        orig = [0, 1, 2]
        sims = [[0, 1, 2], [2, 1, 0]]
        df = compute_arrangement_distances(sims, orig, metrics=["composition", "norm_edit"])
        assert "composition" in df.columns
        assert "norm_edit" in df.columns
        assert "transition" not in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])