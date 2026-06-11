"""Tests for paired Wilcoxon signed-rank test."""

import numpy as np
import pytest
from mc_regime.inference.paired_test import paired_loss_test


class TestPairedLossTest:
    def test_detects_known_difference(self):
        """Small loss_a vs large loss_b should give p < 0.05."""
        np.random.seed(42)
        # Method A has consistently lower losses
        loss_a = np.random.normal(0.5, 0.1, 20)
        loss_b = np.random.normal(1.0, 0.1, 20)

        result = paired_loss_test(loss_a, loss_b, alternative="less")

        assert result["p_value"] < 0.05, f"Expected p < 0.05, got {result['p_value']}"

    def test_no_difference_detected(self):
        """Similar distributions should give p > 0.05."""
        np.random.seed(42)
        # Both methods have similar losses
        loss_a = np.random.normal(1.0, 0.1, 30)
        loss_b = np.random.normal(1.0, 0.1, 30)

        result = paired_loss_test(loss_a, loss_b, alternative="less")

        assert result["p_value"] > 0.05, f"Expected p > 0.05, got {result['p_value']}"

    def test_correct_effect_size_sign(self):
        """When a < b, effect size should be negative."""
        np.random.seed(42)
        loss_a = np.array([1.0, 1.1, 1.0, 0.9, 1.0])
        loss_b = np.array([2.0, 2.1, 2.0, 1.9, 2.0])

        result = paired_loss_test(loss_a, loss_b, alternative="less")

        # Since loss_a < loss_b, differences are negative, so effect size should be negative
        assert result["effect_size"] < 0, f"Expected negative effect size, got {result['effect_size']}"

    def test_identical_arrays_returns_one(self):
        """Identical losses should give p=1.0."""
        loss = np.ones(10)

        result = paired_loss_test(loss, loss, alternative="less")

        assert result["p_value"] == 1.0

    def test_mismatched_shapes_raises(self):
        """Different shaped arrays should raise ValueError."""
        loss_a = np.ones(10)
        loss_b = np.ones(20)

        with pytest.raises(ValueError):
            paired_loss_test(loss_a, loss_b, alternative="less")


class TestPairedLossTestReturnStructure:
    def test_has_required_keys(self):
        """Result should contain all required keys."""
        loss_a = np.ones(10)
        loss_b = np.ones(10) * 2

        result = paired_loss_test(loss_a, loss_b, alternative="less")

        assert "p_value" in result
        assert "statistic" in result
        assert "effect_size" in result
        assert "median_difference" in result

    def test_p_value_in_valid_range(self):
        """p-value should be in [0, 1]."""
        np.random.seed(42)
        loss_a = np.random.normal(0.5, 0.1, 20)
        loss_b = np.random.normal(1.0, 0.1, 20)

        result = paired_loss_test(loss_a, loss_b, alternative="less")

        assert 0.0 <= result["p_value"] <= 1.0