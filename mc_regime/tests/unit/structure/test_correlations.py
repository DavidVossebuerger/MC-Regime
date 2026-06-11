"""Unit tests for correlation bias (Level 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestComputeCorrelations:
    """Tests for compute_correlations function."""

    def test_basic_correlation(self):
        """Basic correlation computation works."""
        data = pd.DataFrame({
            "X": [1, 2, 3, 4, 5],
            "Y": [2, 4, 6, 8, 10],
        })
        from mc_regime.structure.correlations import compute_correlations

        result = compute_correlations(data, pairs=[("X", "Y")])
        assert abs(result[("X", "Y")] - 1.0) < 1e-6  # Perfect correlation

    def test_negative_correlation(self):
        """Negative correlation is detected."""
        data = pd.DataFrame({
            "X": [1, 2, 3, 4, 5],
            "Y": [10, 8, 6, 4, 2],
        })
        from mc_regime.structure.correlations import compute_correlations

        result = compute_correlations(data, pairs=[("X", "Y")])
        assert result[("X", "Y")] < -0.9

    def test_missing_column_raises(self):
        """Missing column logs warning and skips."""
        import warnings
        data = pd.DataFrame({"X": [1, 2, 3]})
        from mc_regime.structure.correlations import compute_correlations

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_correlations(data, pairs=[("X", "Y")])
            # Should skip and return empty result (or partial)
            assert len(w) > 0  # Warning issued


class TestCorrelationBias:
    """Tests for correlation bias computation."""

    def test_zero_bias_identical(self):
        """Zero bias when bootstrap = original."""
        original = {("X", "Y"): 0.5}
        # Original value copied exactly - no bias
        boot_dists = [
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.5},
        ]
        from mc_regime.structure.correlations import compute_correlation_bias

        bias = compute_correlation_bias(original, boot_dists)
        assert abs(bias[("X", "Y")]) < 1e-6

    def test_positive_bias(self):
        """Positive bias when bootstrap mean > original."""
        original = {("X", "Y"): 0.3}
        boot_dists = [
            {("X", "Y"): 0.4},
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.6},
        ]
        from mc_regime.structure.correlations import compute_correlation_bias

        bias = compute_correlation_bias(original, boot_dists)
        assert bias[("X", "Y")] > 0  # 0.5 - 0.3 = 0.2

    def test_default_pairs(self):
        """Default correlation pairs are defined."""
        from mc_regime.structure.correlations import DEFAULT_CORRELATION_PAIRS

        # Check we have 9 pairs as specified
        assert len(DEFAULT_CORRELATION_PAIRS) == 9


class TestL1Bias:
    """Tests for L1 bias metric."""

    def test_l1_zero_for_identical(self):
        """L1 = 0 when all replicates identical."""
        original = {("X", "Y"): 0.5}
        boot_dists = [
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.5},
        ]
        from mc_regime.structure.correlations import compute_l1_bias

        l1 = compute_l1_bias(boot_dists, original)
        assert l1 < 1e-6

    def test_l1_nonzero_for_shifted(self):
        """L1 > 0 when bootstrap is shifted."""
        original = {("X", "Y"): 0.3}
        boot_dists = [
            {("X", "Y"): 0.4},
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.6},
        ]
        from mc_regime.structure.correlations import compute_l1_bias

        l1 = compute_l1_bias(boot_dists, original)
        assert l1 > 0.1  # Expected: mean(0.5) - 0.3 = 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])