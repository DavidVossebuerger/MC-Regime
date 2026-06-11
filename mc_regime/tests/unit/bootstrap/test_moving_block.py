"""Tests for Moving Block Bootstrap sampler."""

import numpy as np
import pandas as pd

from mc_regime.bootstrap.samplers.moving_block import MovingBlockSampler


def _generate_ar1_data(n: int = 200, phi: float = 0.7, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic AR(1) time series for testing."""
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(0, 1, n)
    y = np.zeros(n)
    y[0] = epsilon[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + epsilon[t]
    return pd.DataFrame({"value": y})


def _generate_monthly_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate real-looking monthly FX returns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="ME")
    returns = rng.normal(0.001, 0.02, n)
    return pd.DataFrame({"date": dates, "return": returns})


class TestMovingBlockSampler:
    """Test cases for Moving Block Bootstrap sampler."""

    def test_preserves_shape(self):
        """MBB should preserve shape of original data."""
        data = _generate_monthly_data(200)
        sampler = MovingBlockSampler()

        result = sampler.sample(data, 200, np.random.default_rng(42))

        assert len(result) == 200

    def test_default_block_length_t_1_3(self):
        """Default block length should be T^(1/3)."""
        data = _generate_monthly_data(200)
        sampler = MovingBlockSampler()

        block_len = sampler._compute_block_length(200)

        expected = max(1, int(round(200 ** (1 / 3))))
        assert block_len == expected

    def test_custom_block_length(self):
        """Custom block length should override default."""
        sampler = MovingBlockSampler(block_length=10)

        block_len = sampler._compute_block_length(200)

        assert block_len == 10

    def test_preserves_local_autocorrelation(self):
        """MBB should preserve local autocorrelation better than IID."""
        data = _generate_ar1_data(200, phi=0.7, seed=42)
        sampler = MovingBlockSampler()

        resampled = sampler.sample(data, 200, np.random.default_rng(123))

        # We expect some autocorrelation preserved for nearby lags
        values = resampled["value"].values
        n = len(values)
        if n > 1:
            # Lag-1 correlation
            corr = np.corrcoef(values[:-1], values[1:])[0, 1]
            # Should retain some positive correlation
            assert corr > 0

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        data = _generate_monthly_data(200, seed=42)
        sampler = MovingBlockSampler()

        result1 = sampler.sample(data, 200, np.random.default_rng(42))
        result2 = sampler.sample(data, 200, np.random.default_rng(42))

        pd.testing.assert_frame_equal(result1, result2)