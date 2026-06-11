"""Unit tests for Wasserstein distance (Level 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestWassersteinDistances:
    """Tests for Wasserstein distance computation."""

    def test_zero_distance_point_mass_identical(self):
        """Zero W1 for point-mass when boot = original."""
        original = {("X", "Y"): 0.5}
        boot_dists = [
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.5},
        ]
        from mc_regime.structure.wasserstein import compute_wasserstein_distances

        w_dist = compute_wasserstein_distances(original, boot_dists)
        assert w_dist[("X", "Y")] < 1e-6

    def test_positive_distance_shifted(self):
        """Positive W1 for shifted distribution."""
        original = {("X", "Y"): 0.5}
        # All bootstrap values are 0.6
        boot_dists = [
            {("X", "Y"): 0.6},
            {("X", "Y"): 0.6},
            {("X", "Y"): 0.6},
        ]
        from mc_regime.structure.wasserstein import compute_wasserstein_distances

        w_dist = compute_wasserstein_distances(original, boot_dists)
        # W1 = mean(|0.6 - 0.5|) = 0.1
        assert abs(w_dist[("X", "Y")] - 0.1) < 1e-6

    def test_varied_bootstrap_distribution(self):
        """Correct W1 for varied bootstrap distribution."""
        original = {("X", "Y"): 0.5}
        # Values: 0.5, 0.6, 0.7 -> mean absolute dev = |0|+|0.1|+|0.2|/3 = 0.1
        boot_dists = [
            {("X", "Y"): 0.5},
            {("X", "Y"): 0.6},
            {("X", "Y"): 0.7},
        ]
        from mc_regime.structure.wasserstein import compute_wasserstein_distances

        w_dist = compute_wasserstein_distances(original, boot_dists)
        expected = (abs(0.5 - 0.5) + abs(0.6 - 0.5) + abs(0.7 - 0.5)) / 3
        assert abs(w_dist[("X", "Y")] - expected) < 1e-6

    def test_multiple_pairs(self):
        """Works correctly with multiple pairs."""
        original = {("X", "Y"): 0.5, ("A", "B"): 0.3}
        boot_dists = [
            {("X", "Y"): 0.5, ("A", "B"): 0.3},
            {("X", "Y"): 0.6, ("A", "B"): 0.4},
        ]
        from mc_regime.structure.wasserstein import compute_wasserstein_distances

        w_dist = compute_wasserstein_distances(original, boot_dists)
        assert len(w_dist) == 2
        assert w_dist[("X", "Y")] > 0
        assert w_dist[("A", "B")] > 0


class TestL2Wasserstein:
    """Tests for L2 metric."""

    def test_l2_zero_for_identical(self):
        """L2 = 0 for identical bootstrap."""
        original = {("X", "Y"): 0.5}
        boot_dists = [{("X", "Y"): 0.5}] * 5
        from mc_regime.structure.wasserstein import compute_l2_wasserstein

        l2 = compute_l2_wasserstein(boot_dists, original)
        assert l2 < 1e-6

    def test_l2_positive_for_shifted(self):
        """L2 > 0 for shifted distribution."""
        original = {("X", "Y"): 0.5, ("A", "B"): 0.3}
        boot_dists = [
            {("X", "Y"): 0.6, ("A", "B"): 0.4},
            {("X", "Y"): 0.6, ("A", "B"): 0.4},
        ]
        from mc_regime.structure.wasserstein import compute_l2_wasserstein

        l2 = compute_l2_wasserstein(boot_dists, original)
        # Pair X,Y: |0.6-0.5| = 0.1
        # Pair A,B: |0.4-0.3| = 0.1
        # Mean = 0.1
        assert abs(l2 - 0.1) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])