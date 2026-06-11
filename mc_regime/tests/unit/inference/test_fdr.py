"""Tests for Benjamini-Hochberg FDR correction."""

import numpy as np
import pytest
from mc_regime.inference.fdr import apply_bh_fdr, bh_reject_hypotheses


class TestApplyBHFDR:
    def test_known_p_values_adjusted(self):
        """Known p-values should produce expected adjusted q-values."""
        # Classic BH example: p-values at standard thresholds
        # Sorted: 0.01 <= 0.02 <= 0.03 <= 0.04 <= 0.10 <= 0.40 <= 0.60 <= 0.80 <= 0.90 <= 1.00
        # Critical values at alpha=0.10: i/m * 0.10 = 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
        # Reject if p <= critical: 0.01<=0.01(Y), 0.02<=0.02(Y), 0.03<=0.03(Y), 0.04<=0.04(Y), 0.10<=0.05(N)
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.10, 0.40, 0.60, 0.80, 0.90, 1.00])
        alpha = 0.10

        q_values = apply_bh_fdr(p_values, alpha=alpha)

        # First 4 should be rejected
        expected_rejected = np.array([True, True, True, True, False, False, False, False, False, False])
        actual_rejected = q_values <= alpha

        np.testing.assert_array_equal(actual_rejected, expected_rejected)

    def test_q_values_monotonic(self):
        """Q-values should be monotonically increasing."""
        p_values = np.array([0.001, 0.01, 0.05, 0.10, 0.30, 0.50])

        q_values = apply_bh_fdr(p_values, alpha=0.10)

        assert np.all(np.diff(q_values) >= -1e-10)

    def test_q_values_ge_original(self):
        """Q-values should be >= original p-values."""
        np.random.seed(42)
        p_values = np.random.uniform(0.01, 0.5, 20)

        q_values = apply_bh_fdr(p_values, alpha=0.10)

        assert np.all(q_values >= p_values - 1e-10)

    def test_empty_array(self):
        """Empty array should return empty array."""
        p_values = np.array([])

        q_values = apply_bh_fdr(p_values, alpha=0.10)

        assert len(q_values) == 0

    def test_invalid_p_values_raises(self):
        """P-values outside [0,1] should raise ValueError."""
        p_values = np.array([-0.1, 0.5, 0.9])

        with pytest.raises(ValueError):
            apply_bh_fdr(p_values, alpha=0.10)

    def test_pgt1_raises(self):
        """P-values > 1 should raise ValueError."""
        p_values = np.array([0.5, 1.1, 0.9])

        with pytest.raises(ValueError):
            apply_bh_fdr(p_values, alpha=0.10)


class TestBHRejectHypotheses:
    def test_expected_rejections(self):
        """Should reject correct hypotheses."""
        # With sorted p-values: 0.001 < 0.005 < 0.010 < 0.100 < 0.500
        # Critical values (i/5 * 0.10): 0.02, 0.04, 0.06, 0.08, 0.10
        # Reject if p <= critical: 0.001<=0.02(Y), 0.005<=0.04(Y), 0.010<=0.06(Y), 0.100<=0.08(N), 0.500<=0.10(N)
        p_values = np.array([0.001, 0.005, 0.010, 0.100, 0.500])
        alpha = 0.10

        rejected = bh_reject_hypotheses(p_values, alpha=alpha)

        expected = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(rejected, expected)


class TestApplyBHFDRIntegration:
    def test_all_high_pvalues(self):
        """All high p-values -> none rejected."""
        p_values = np.array([0.9, 0.95, 1.0])

        q_values = apply_bh_fdr(p_values, alpha=0.10)

        assert np.all(q_values > 0.10)

    def test_all_low_pvalues(self):
        """All low p-values -> all rejected."""
        p_values = np.array([0.001, 0.002, 0.003])

        q_values = apply_bh_fdr(p_values, alpha=0.10)

        assert np.all(q_values <= 0.10)