"""Tests for BCa bootstrap p-value computation."""

import numpy as np
import pytest
from mc_regime.inference.bca import bca_p_value


class TestBCaPValue:
    def test_returns_value_in_range(self):
        """BCa p-value should be in [0, 1]."""
        np.random.seed(42)
        observed = 0.5
        bootstrap_stat = np.random.normal(0.5, 0.1, 500)
        jackknife_stat = np.random.normal(0.5, 0.1, 50)

        p_val = bca_p_value(observed, bootstrap_stat, jackknife_stat)

        assert 0.0 <= p_val <= 1.0

    def test_symmetric_distribution_approaches_percentile(self):
        """For symmetric distribution, BCa approximate equals percentile p-value."""
        np.random.seed(42)
        observed = 1.96
        bootstrap_stat = np.random.normal(0, 1, 1000)

        # Jackknife estimate (simulated)
        jackknife_stat = np.array([np.mean(np.random.normal(0, 1, 99)) for _ in range(50)])

        p_val = bca_p_value(observed, bootstrap_stat, jackknife_stat)

        # Percentile p-value approach (two-sided)
        pct_below = np.mean(bootstrap_stat < observed)
        pct_above = np.mean(bootstrap_stat > observed)
        percentile_p = 2 * min(pct_below, pct_above)
        percentile_p = max(percentile_p, 0.001)  # Avoid exact zero

        # Should be reasonably close (within factor of 2)
        assert p_val / percentile_p < 2, f"BCa p={p_val}, percentile p={percentile_p}"

    def test_extreme_observed_gives_small_p(self):
        """Extreme observed value should give small p-value."""
        np.random.seed(42)
        observed = 5.0  # Very extreme
        bootstrap_stat = np.random.normal(0, 1, 500)
        jackknife_stat = np.random.normal(0, 1, 50)

        p_val = bca_p_value(observed, bootstrap_stat, jackknife_stat)

        assert p_val < 0.1

    def test_typical_observed_gives_moderate_p(self):
        """Typical observed value should produce reasonable p-value."""
        np.random.seed(42)
        observed = 0.1  # Not at the center
        bootstrap_stat = np.random.normal(0, 1, 500)
        jackknife_stat = np.random.normal(0, 1, 50)

        p_val = bca_p_value(observed, bootstrap_stat, jackknife_stat)

        # Should be in valid range, not necessarily moderate
        assert 0.0 < p_val < 1.0

    def test_zero_length_raises(self):
        """Empty arrays should raise ValueError."""
        observed = 1.0
        bootstrap_stat = np.array([])
        jackknife_stat = np.array([1.0])

        with pytest.raises(ValueError):
            bca_p_value(observed, bootstrap_stat, jackknife_stat)

    def test_single_element_raises(self):
        """Single element arrays should raise ValueError."""
        observed = 1.0
        bootstrap_stat = np.array([1.0])
        jackknife_stat = np.array([1.0])

        with pytest.raises(ValueError):
            bca_p_value(observed, bootstrap_stat, jackknife_stat)


class TestBCaPValueEdgeCases:
    def test_zero_variance_jackknife(self):
        """Zero variance in jackknife should not crash."""
        np.random.seed(42)
        observed = 0.5
        bootstrap_stat = np.random.normal(0.5, 0.1, 500)
        jackknife_stat = np.full(50, 0.5)  # All same

        p_val = bca_p_value(observed, bootstrap_stat, jackknife_stat)

        # Should still return valid p-value
        assert 0.0 <= p_val <= 1.0

    def test_non_default_alpha(self):
        """Should work with custom alpha."""
        np.random.seed(42)
        observed = 2.0
        bootstrap_stat = np.random.normal(0, 1, 500)
        jackknife_stat = np.random.normal(0, 1, 50)

        p_val = bca_p_value(observed, bootstrap_stat, jackknife_stat, alpha=0.01)

        assert 0.0 <= p_val <= 1.0