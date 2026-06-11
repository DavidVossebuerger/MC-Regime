"""Tests for IID bootstrap sampler."""

import numpy as np
import pandas as pd
import pytest

from mc_regime.bootstrap.samplers.iid import IIDSampler


def _autocorr(x, lags):
    """Calculate autocorrelation at given lags."""
    x = np.asarray(x)
    x = x - x.mean()
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(acf) // 2:]
    acf /= acf[0]
    return [acf[lag] for lag in lags]


def generate_ar1_data(n: int = 200, phi: float = 0.7, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic AR(1) time series for testing."""
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(0, 1, n)
    y = np.zeros(n)
    y[0] = epsilon[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + epsilon[t]
    return pd.DataFrame({"value": y})


def generate_monthly_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate real-looking monthly FX returns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="ME")
    returns = rng.normal(0.001, 0.02, n)  # ~2% monthly vol
    return pd.DataFrame({"date": dates, "return": returns})


class TestIIDSampler:
    """Test cases for IID bootstrap sampler."""

    def test_preserves_shape(self):
        """IID should preserve shape of original data."""
        data = generate_monthly_data(200)
        sampler = IIDSampler()

        result = sampler.sample(data, 200, np.random.default_rng(42))

        assert len(result) == 200
        assert list(result.columns) == ["date", "return"]

    def test_preserves_marginal_distribution(self):
        """IID should preserve marginal distribution approximately."""
        data = generate_monthly_data(200, seed=42)
        sampler = IIDSampler()

        # Many replications to test distribution preservation
        results = [sampler.sample(data, 200, np.random.default_rng(i)) for i in range(100)]

        # Sample mean should be close to original
        original_mean = data["return"].mean()
        sample_means = [r["return"].mean() for r in results]
        mean_of_means = np.mean(sample_means)

        # Should be close to original mean
        assert abs(mean_of_means - original_mean) < 0.01

    def test_breaks_autocorrelation(self):
        """IID should break autocorrelation in AR(1) data."""
        data = generate_ar1_data(200, phi=0.7, seed=42)

        # Verify original has autocorrelation
        orig_acf = _autocorr(data["value"].values, [1])[0]
        assert abs(orig_acf) > 0.3, "Original should have autocorrelation"

        sampler = IIDSampler()
        resampled = sampler.sample(data, 200, np.random.default_rng(123))

        # Compute autocorrelation of resampled
        resampled_acf = _autocorr(resampled["value"].values, [1])[0]

        # Resampled autocorrelation should be much smaller
        assert abs(resampled_acf) < 0.2, "IID should break autocorrelation"

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        data = generate_monthly_data(200, seed=42)
        sampler = IIDSampler()

        result1 = sampler.sample(data, 200, np.random.default_rng(42))
        result2 = sampler.sample(data, 200, np.random.default_rng(42))

        pd.testing.assert_frame_equal(result1, result2)