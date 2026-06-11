"""Unit tests for network structure (Level 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestBuildCorrelationGraph:
    """Tests for building correlation adjacency matrix."""

    def test_symmetric_matrix(self):
        """Matrix is symmetric."""
        data = pd.DataFrame({
            "X": np.random.randn(100),
            "Y": np.random.randn(100),
            "Z": np.random.randn(100),
        })
        from mc_regime.structure.network import build_correlation_graph

        A = build_correlation_graph(data, nodes=["X", "Y", "Z"])
        assert np.allclose(A, A.T)

    def test_diagonal_unit(self):
        """Diagonal is 1.0."""
        data = pd.DataFrame({
            "X": np.random.randn(100),
            "Y": np.random.randn(100),
        })
        from mc_regime.structure.network import build_correlation_graph

        A = build_correlation_graph(data, nodes=["X", "Y"])
        assert np.allclose(np.diag(A), 1.0)

    def test_perfect_correlation(self):
        """Perfect correlation gives |corr| = 1.0."""
        data = pd.DataFrame({
            "X": [1, 2, 3, 4, 5],
            "Y": [2, 4, 6, 8, 10],
        })
        from mc_regime.structure.network import build_correlation_graph

        A = build_correlation_graph(data, nodes=["X", "Y"])
        assert abs(A[0, 1] - 1.0) < 1e-6

    def test_absolute_metric(self):
        """Absolute metric uses |corr|."""
        data = pd.DataFrame({
            "X": [1, 2, 3, 4, 5],
            "Y": [-1, -2, -3, -4, -5],
        })
        from mc_regime.structure.network import build_correlation_graph

        A_abs = build_correlation_graph(data, nodes=["X", "Y"], metric="absolute")
        A_sig = build_correlation_graph(data, nodes=["X", "Y"], metric="signed")

        assert abs(A_abs[0, 1] - 1.0) < 1e-6  # |corr| = 1
        assert abs(A_sig[0, 1] - (-1.0)) < 1e-6  # corr = -1


class TestNetworkDistance:
    """Tests for network distance computation."""

    def test_frobenius_zero_identical(self):
        """Frobenius distance is zero for identical matrices."""
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        from mc_regime.structure.network import compute_network_distance

        dist = compute_network_distance(A, A, metric="frobenius")
        assert dist < 1e-6

    def test_frobenius_positive_different(self):
        """Frobenius > 0 for different matrices."""
        A1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        A2 = np.array([[1.0, 0.3], [0.3, 1.0]])
        from mc_regime.structure.network import compute_network_distance

        dist = compute_network_distance(A1, A2, metric="frobenius")
        assert dist > 0

    def test_spectral_zero_identical(self):
        """Spectral distance is zero for identical matrices."""
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        from mc_regime.structure.network import compute_network_distance

        dist = compute_network_distance(A, A, metric="spectral")
        assert dist < 1e-6

    def test_spectral_positive_different(self):
        """Spectral > 0 for different matrices."""
        A1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        A2 = np.array([[1.0, 0.3], [0.3, 1.0]])
        from mc_regime.structure.network import compute_network_distance

        dist = compute_network_distance(A1, A2, metric="spectral")
        assert dist > 0


class TestL3Network:
    """Tests for L3 network metrics."""

    def test_l3_zero_identical(self):
        """L3 = 0 when bootstrap = original."""
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        boot_graphs = [A.copy(), A.copy(), A.copy()]
        from mc_regime.structure.network import compute_l3_network

        l3 = compute_l3_network(A, boot_graphs)
        assert l3["frobenius"] < 1e-6
        assert l3["spectral"] < 1e-6

    def test_l3_both_metrics(self):
        """Both frobenius and spectral metrics returned."""
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        boot_graphs = [
            np.array([[1.0, 0.4], [0.4, 1.0]]),
            np.array([[1.0, 0.6], [0.6, 1.0]]),
        ]
        from mc_regime.structure.network import compute_l3_network

        l3 = compute_l3_network(A, boot_graphs, metrics=("frobenius", "spectral"))
        assert "frobenius" in l3
        assert "spectral" in l3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])