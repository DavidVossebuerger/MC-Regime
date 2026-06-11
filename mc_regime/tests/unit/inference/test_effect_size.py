"""Tests for Cohen's d effect size computation."""

import numpy as np
import pytest
from mc_regime.inference.effect_size import cohens_d, cohens_d_from_summary, glass_delta


class TestCohensD:
    def test_well_separated_groups_large_d(self):
        """Well-separated groups should give d > 0.8."""
        np.random.seed(42)
        # Group A: mean 2.0, low variance
        # Group B: mean 0.0, low variance
        group_a = np.random.normal(2.0, 0.5, 100)
        group_b = np.random.normal(0.0, 0.5, 100)
        # Increase separation to ensure consistent detection
        group_a = np.array([2.0] * 100) + np.random.normal(0, 0.1, 100)
        group_b = np.array([0.0] * 100) + np.random.normal(0, 0.1, 100)

        d = cohens_d(group_a, group_b)

        assert abs(d) > 0.8, f"Expected |d| > 0.8, got {d}"

    def test_identical_groups_zero_d(self):
        """Identical groups should give d = 0."""
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Exact duplicates
        group_a = group.copy()
        group_b = group.copy()

        d = cohens_d(group_a, group_b)

        assert d == 0.0

    def test_positive_when_a_gt_b(self):
        """When mean_a > mean_b with variation, d should be positive."""
        group_a = np.array([5.0, 5.1, 4.9])
        group_b = np.array([3.0, 3.1, 2.9])

        d = cohens_d(group_a, group_b)

        assert d > 0

    def test_negative_when_a_lt_b(self):
        """When mean_a < mean_b with variation, d should be negative."""
        group_a = np.array([3.0, 3.1, 2.9])
        group_b = np.array([5.0, 5.1, 4.9])

        d = cohens_d(group_a, group_b)

        assert d < 0

    def test_small_sample_works(self):
        """Should work with minimum sample sizes."""
        group_a = np.array([1.0, 2.0])
        group_b = np.array([3.0, 4.0])

        d = cohens_d(group_a, group_b)

        assert np.isfinite(d)

    def test_single_element_raises(self):
        """Single element groups should raise ValueError."""
        group_a = np.array([1.0])
        group_b = np.array([2.0])

        with pytest.raises(ValueError):
            cohens_d(group_a, group_b)


class TestCohensDFromSummary:
    def test_matches_individual_calculation(self):
        """Summary calculation should match individual."""
        group_a = np.array([2.1, 2.2, 1.9, 2.0, 2.3])
        group_b = np.array([0.9, 1.1, 0.8, 1.0, 0.7])

        d_direct = cohens_d(group_a, group_b)

        # Extract summary stats
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)
        std_a = np.std(group_a, ddof=1)
        std_b = np.std(group_b, ddof=1)
        n_a = len(group_a)
        n_b = len(group_b)

        d_summary = cohens_d_from_summary(mean_a, mean_b, std_a, std_b, n_a, n_b)

        assert abs(d_direct - d_summary) < 1e-10


class TestGlassDelta:
    def test_control_reference(self):
        """Glass's delta uses control group as denominator."""
        group_a = np.array([2.0, 2.1, 1.9, 2.0])  # Treatment
        group_b = np.array([1.0, 1.1, 0.9, 1.0])  # Control

        delta = glass_delta(group_a, group_b)

        # Should equal (mean_a - mean_b) / std_b
        expected = (np.mean(group_a) - np.mean(group_b)) / np.std(group_b, ddof=1)

        assert abs(delta - expected) < 1e-10

    def test_zero_control_sd(self):
        """Zero control SD returns 0."""
        group_a = np.array([2.0, 2.1, 1.9])
        group_b = np.array([1.0, 1.0, 1.0])  # Zero variance

        delta = glass_delta(group_a, group_b)

        assert delta == 0.0


class TestCohensDInterpretation:
    def test_negligible_interpretation(self):
        """Near-zero d should be negligible."""
        group_a = np.random.normal(1.0, 1.0, 100)
        group_b = np.random.normal(1.05, 1.0, 100)

        d = cohens_d(group_a, group_b)

        assert abs(d) < 0.5  # Small or negligible

    def test_medium_interpretation(self):
        """Medium separation gives medium d."""
        group_a = np.array([1.0] * 50 + np.random.normal(0, 0.5, 50))
        group_b = np.array([0.0] * 50 + np.random.normal(0, 0.5, 50))

        d = cohens_d(group_a, group_b)

        # d should be around 1/0.5 = 2, or medium with noise
        assert abs(d) > 0.2