"""Tests for Regime-Preserving Bootstrap sampler."""

import numpy as np
import pandas as pd
import pytest

from mc_regime.bootstrap.blocks import Block
from mc_regime.bootstrap.samplers.regime import RegimeSampler


def _generate_monthly_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate real-looking monthly FX returns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="ME")
    returns = rng.normal(0.001, 0.02, n)
    return pd.DataFrame({"date": dates, "return": returns})


def _create_test_blocks(n: int = 200, n_regimes: int = 3) -> list[Block]:
    """Create test regime blocks with equal lengths."""
    block_size = n // n_regimes
    blocks = []
    for i in range(n_regimes):
        start = i * block_size
        end = min((i + 1) * block_size - 1, n - 1)
        blocks.append(Block(start_idx=start, end_idx=end, regime_label=i))
    return blocks


class TestRegimeSampler:
    """Test cases for Regime-Preserving Bootstrap sampler."""

    def test_preserves_shape(self):
        """Regime should preserve shape of original data."""
        data = _generate_monthly_data(200)
        sampler = RegimeSampler(mode="uniform")
        blocks = _create_test_blocks(200)

        result = sampler.sample(data, 200, np.random.default_rng(42), blocks=blocks)

        assert len(result) == 200

    def test_uniform_mode(self):
        """Uniform mode should pick random blocks."""
        data = _generate_monthly_data(200)
        sampler = RegimeSampler(mode="uniform")
        blocks = _create_test_blocks(200)

        # Same seed should give same result
        result1 = sampler.sample(data, 200, np.random.default_rng(42), blocks=blocks)
        result2 = sampler.sample(data, 200, np.random.default_rng(42), blocks=blocks)

        pd.testing.assert_frame_equal(result1, result2)

    def test_requires_transition_matrix_for_markov(self):
        """Markov mode requires transition matrix."""
        with pytest.raises(ValueError):
            RegimeSampler(mode="markov")

    def test_markov_mode_accepts_transition_matrix(self):
        """Markov mode should accept valid transition matrix."""
        trans = np.array([[0.7, 0.3], [0.2, 0.8]])
        sampler = RegimeSampler(mode="markov", transition_matrix=trans)

        assert sampler.mode == "markov"
        assert sampler.transition_matrix is not None

    def test_handles_empty_blocks(self):
        """Should fall back to whole series for empty blocks."""
        data = _generate_monthly_data(200)
        sampler = RegimeSampler(mode="uniform")

        # No blocks provided - should treat as single block
        result = sampler.sample(data, 200, np.random.default_rng(42), blocks=None)

        assert len(result) == 200

    def test_respects_regime_boundaries(self):
        """Resampled data should respect regime block boundaries."""
        data = _generate_monthly_data(200)
        sampler = RegimeSampler(mode="uniform")
        blocks = _create_test_blocks(200)

        # Sample larger than original to test truncation
        result = sampler.sample(data, 250, np.random.default_rng(42), blocks=blocks)

        # Should only have full blocks, truncated at end
        assert len(result) <= 250

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        data = _generate_monthly_data(200, seed=42)
        sampler = RegimeSampler(mode="uniform")
        blocks = _create_test_blocks(200)

        result1 = sampler.sample(data, 200, np.random.default_rng(42), blocks=blocks)
        result2 = sampler.sample(data, 200, np.random.default_rng(42), blocks=blocks)

        pd.testing.assert_frame_equal(result1, result2)