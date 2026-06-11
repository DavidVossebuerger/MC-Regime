"""Unit tests for composite loss (supplementary metric)."""
from __future__ import annotations

import numpy as np
import pytest


class TestCompositeLoss:
    """Tests for composite loss computation."""

    def test_default_weights_equal(self):
        """Default weights are equal (0.25 each)."""
        from mc_regime.structure.composite import compute_composite_loss

        composite = compute_composite_loss(0.1, 0.2, 0.3, 0.4)
        expected = (0.25 * 0.1 + 0.25 * 0.2 + 0.25 * 0.3 + 0.25 * 0.4)
        assert abs(composite - expected) < 1e-6

    def test_custom_weights(self):
        """Custom weights work."""
        from mc_regime.structure.composite import compute_composite_loss

        composite = compute_composite_loss(
            0.1, 0.2, 0.3, 0.4,
            weights=(0.5, 0.3, 0.1, 0.1)
        )
        expected = (0.5 * 0.1 + 0.3 * 0.2 + 0.1 * 0.3 + 0.1 * 0.4) / 1.0
        assert abs(composite - expected) < 1e-6

    def test_manual_calculation(self):
        """Matches manual calculation."""
        from mc_regime.structure.composite import compute_composite_loss

        L1, L2, L3_fro, L3_spec = 0.1, 0.15, 0.2, 0.25
        weights = (0.25, 0.25, 0.25, 0.25)

        composite = compute_composite_loss(L1, L2, L3_fro, L3_spec, weights)
        manual = (L1 + L2 + L3_fro + L3_spec) / 4

        assert abs(composite - manual) < 1e-6


class TestStructurePreservationReport:
    """Tests for report generation."""

    def test_report_contains_all_fields(self):
        """Report dictionary has all fields."""
        from mc_regime.structure.composite import structure_preservation_report

        report = structure_preservation_report(0.1, 0.2, 0.3, 0.4)

        assert "L1_correlation_bias" in report
        assert "L2_wasserstein" in report
        assert "L3_frobenius" in report
        assert "L3_spectral" in report
        assert "composite" in report

    def test_composite_included_if_none(self):
        """Composite computed if not provided."""
        from mc_regime.structure.composite import structure_preservation_report

        report = structure_preservation_report(0.1, 0.2, 0.3, 0.4)
        # With default weights (0.25), L = mean of all 4
        expected_composite = (0.1 + 0.2 + 0.3 + 0.4) / 4
        assert abs(report["composite"] - expected_composite) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])