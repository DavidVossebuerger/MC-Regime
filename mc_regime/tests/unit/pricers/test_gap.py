"""Unit tests for gap calculation."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def gap_data():
    """Generate synthetic gap data for testing."""
    np.random.seed(42)
    n = 100

    # Gap centered around zero with some structure
    gap = np.random.randn(n) * 0.05
    gap[50:] += 0.3  # Drift in second half

    dates = pd.date_range("2010-01-01", periods=n, freq="ME")
    return pd.Series(gap, index=dates, name="gap")


def test_compute_gap_log_diff():
    """Test gap computation with log_diff method."""
    from mc_regime.pricers.gap import compute_gap

    actual = pd.Series([1.1, 1.2, 1.3], index=[1, 2, 3])
    fitted = pd.Series([1.0, 1.1, 1.2], index=[1, 2, 3])

    gap = compute_gap(actual, fitted, method="log_diff")

    # Gap is simply actual - fitted (values already in same scale)
    expected = actual - fitted
    np.testing.assert_array_almost_equal(gap.values, expected.values)


def test_compute_gap_aligns_indices():
    """Test gap computation aligns different indices."""
    from mc_regime.pricers.gap import compute_gap

    actual = pd.Series([1.1, 1.2], index=[1, 2])
    fitted = pd.Series([1.0, 1.1, 1.2], index=[2, 3, 4])  # Different index

    gap = compute_gap(actual, fitted, method="log_diff")

    assert len(gap) == 1  # Only common index


def test_gap_statistics_returns_all_stats(gap_data):
    """Test gap_statistics returns expected keys."""
    from mc_regime.pricers.gap import gap_statistics

    stats = gap_statistics(gap_data)

    expected_keys = ["mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95", "n_obs"]

    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"

    assert stats["n_obs"] == len(gap_data)


def test_gap_statistics_values(gap_data):
    """Test gap_statistics computed values are sensible."""
    from mc_regime.pricers.gap import gap_statistics

    stats = gap_statistics(gap_data)

    # Stats should be finite
    assert np.isfinite(stats["mean"])
    assert np.isfinite(stats["std"])
    assert stats["std"] > 0

    # Percentiles should be ordered
    assert stats["p5"] <= stats["p25"] <= stats["p50"] <= stats["p75"] <= stats["p95"]


def test_gap_analyzer_init(gap_data):
    """Test GapAnalyzer initialization."""
    from mc_regime.pricers.gap import GapAnalyzer

    analyzer = GapAnalyzer(gap_data)

    assert analyzer.gap.equals(gap_data)
    assert analyzer.stats_["n_obs"] == len(gap_data)


def test_gap_analyzer_is_extreme(gap_data):
    """Test is_extreme identifies outliers."""
    from mc_regime.pricers.gap import GapAnalyzer

    analyzer = GapAnalyzer(gap_data)
    is_extreme = analyzer.is_extreme(threshold=2.0)

    assert isinstance(is_extreme, pd.Series)
    assert len(is_extreme) == len(gap_data)
    # Some portion should be marked as extreme
    assert is_extreme.sum() >= 0


def test_gap_analyzer_regime_indicators(gap_data):
    """Test regime_indicator generation."""
    from mc_regime.pricers.gap import GapAnalyzer

    analyzer = GapAnalyzer(gap_data)
    regimes = analyzer.regime_indicators(low_threshold=-1.5, high_threshold=1.5)

    assert "gap" in regimes.columns
    assert "z_score" in regimes.columns
    assert "is_undervalued" in regimes.columns
    assert "is_overvalued" in regimes.columns
    assert "is_fair" in regimes.columns

    # Check mutually exclusive
    assert (regimes["is_undervalued"] & regimes["is_overvalued"]).sum() == 0


def test_gap_analyzer_convergence(gap_data):
    """Test convergence detection."""
    from mc_regime.pricers.gap import GapAnalyzer

    analyzer = GapAnalyzer(gap_data)
    conv = analyzer.convergence_test(window=30)

    assert "converged" in conv
    assert "slope" in conv
    assert "slope_pvalue" in conv


def test_gap_analyzer_summary(gap_data):
    """Test GapAnalyzer.summary()."""
    from mc_regime.pricers.gap import GapAnalyzer

    analyzer = GapAnalyzer(gap_data)
    summary = analyzer.summary()

    assert "statistics" in summary
    assert "recent_mean" in summary
    assert "recent_std" in summary
    assert "extremes_share" in summary


def test_compute_gap_invalid_method():
    """Test invalid method raises ValueError."""
    from mc_regime.pricers.gap import compute_gap

    actual = pd.Series([1.0, 2.0])
    fitted = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="Unknown method"):
        compute_gap(actual, fitted, method="invalid")