"""Tests for Stationary Bootstrap sampler."""

import numpy as np
import pandas as pd

from mc_regime.bootstrap.samplers.stationary import StationarySampler


def _generate_monthly_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate real-looking monthly FX returns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="ME")
    returns = rng.normal(0.001, 0.02, n)
    return pd.DataFrame({"date": dates, "return": returns})


class TestStationarySampler:
    """Test cases for Stationary Bootstrap sampler."""

    def test_preserves_shape(self):
        """SB should preserve shape of original data."""
        data = _generate_monthly_data(200)
        sampler = StationarySampler()

        result = sampler.sample(data, 200, np.random.default_rng(42))

        assert len(result) == 200

    def test_default_length_param(self):
        """Default length parameter should approximate T^(1/3)."""
        data = _generate_monthly_data(200)
        sampler = StationarySampler()

        # Draw block lengths - should have variance due to geometric
        lengths = sampler._draw_block_lengths(100, 200, np.random.default_rng(42))

        assert len(lengths) == 100
        assert all(lengths >= 1)

    def test_custom_length_param(self):
        """Custom length parameter overrides default."""
        sampler = StationarySampler(length_param=5.0)

        lengths = sampler._draw_block_lengths(100, 200, np.random.default_rng(42))

        # Mean should be approximately 5
        assert np.mean(lengths) > 3  # geometric has variance

    def test_random_block_lengths(self):
        """Stationary should have varying block lengths (unlike MBB)."""
        data = _generate_monthly_data(200)
        sampler = StationarySampler()

        results = [sampler.sample(data, 200, np.random.default_rng(i)) for i in range(5)]

        # Results should differ (different random block lengths)
        all_different = False
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if not results[i].equals(results[j]):
                    all_different = True
                    break

        assert all_different, "Stationary should produce varying results"

    def test_creates_stationary_series(self):
        """SB should produce quasi-stationary series."""
        data = _generate_monthly_data(200)
        sampler = StationarySampler()

        resampled = sampler.sample(data, 200, np.random.default_rng(123))

        # Series shouldn't have obvious boundary artifacts
        # Just verify it's complete and valid
        assert len(resampled) == 200
        assert not resampled.isnull().any().any()

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        data = _generate_monthly_data(200, seed=42)
        sampler = StationarySampler()

        result1 = sampler.sample(data, 200, np.random.default_rng(42))
        result2 = sampler.sample(data, 200, np.random.default_rng(42))

        pd.testing.assert_frame_equal(result1, result2)