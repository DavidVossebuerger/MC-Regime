"""Tests for BootstrapEngine."""

import numpy as np
import pandas as pd
import pytest

from mc_regime.bootstrap.blocks import Block
from mc_regime.bootstrap.engine import _derive_seed, BootstrapEngine, METHOD_REGISTRY


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


class TestDeriveSeed:
    """Test cases for seed derivation function."""

    def test_returns_integer(self):
        """Should return integer seed."""
        seed = _derive_seed(42, "IID", 0)
        assert isinstance(seed, int)

    def test_deterministic(self):
        """Same inputs should give same seed."""
        seed1 = _derive_seed(42, "IID", 0)
        seed2 = _derive_seed(42, "IID", 0)
        assert seed1 == seed2

    def test_different_methods_give_different_seeds(self):
        """Different methods with same base/rep should give different seeds."""
        seed1 = _derive_seed(42, "IID", 0)
        seed2 = _derive_seed(42, "MBB", 0)
        assert seed1 != seed2

    def test_different_replications_give_different_seeds(self):
        """Different replications should give different seeds."""
        seed1 = _derive_seed(42, "IID", 0)
        seed2 = _derive_seed(42, "IID", 1)
        assert seed1 != seed2


class TestMethodRegistry:
    """Test cases for METHOD_REGISTRY."""

    def test_has_four_methods(self):
        """Registry should have exactly 4 methods."""
        assert len(METHOD_REGISTRY) == 4

    def test_includes_iid(self):
        """Registry should include IID."""
        assert "IID" in METHOD_REGISTRY

    def test_includes_mbb(self):
        """Registry should include MBB."""
        assert "MBB" in METHOD_REGISTRY

    def test_includes_sb(self):
        """Registry should include SB."""
        assert "SB" in METHOD_REGISTRY

    def test_includes_regime(self):
        """Registry should include Regime."""
        assert "Regime" in METHOD_REGISTRY


class TestBootstrapEngine:
    """Test cases for BootstrapEngine."""

    def test_initialization(self):
        """Engine should initialize with parameters."""
        engine = BootstrapEngine(replications=100, base_seed=123)

        assert engine.replications == 100
        assert engine.base_seed == 123

    def test_default_initialization(self):
        """Engine should have sensible defaults."""
        engine = BootstrapEngine()

        assert engine.replications == 999
        assert engine.base_seed == 42

    def test_run_all_returns_dict(self):
        """run_all should return dictionary."""
        data = _generate_monthly_data(200)
        engine = BootstrapEngine(replications=5, base_seed=42)

        results = engine.run_all(data)

        assert isinstance(results, dict)

    def test_run_all_has_four_methods(self):
        """run_all should return results for all 4 methods."""
        data = _generate_monthly_data(200)
        engine = BootstrapEngine(replications=5, base_seed=42)

        results = engine.run_all(data)

        assert len(results) == 4

    def test_run_all_returns_correct_replication_count(self):
        """run_all should return specified number of replicates."""
        data = _generate_monthly_data(200)
        engine = BootstrapEngine(replications=10, base_seed=42)

        results = engine.run_all(data)

        for method_results in results.values():
            assert len(method_results) == 10

    def test_run_all_preserves_data_length(self):
        """run_all should preserve original data length."""
        data = _generate_monthly_data(200)
        engine = BootstrapEngine(replications=5, base_seed=42)

        results = engine.run_all(data)

        for method_name, replicates in results.items():
            for replicated_df in replicates:
                assert len(replicated_df) == 200

    def test_run_method_single(self):
        """run_method should work for single method."""
        data = _generate_monthly_data(200)
        engine = BootstrapEngine(replications=5, base_seed=42)

        results = engine.run_method("IID", data)

        assert isinstance(results, list)
        assert len(results) == 5

    def test_run_method_invalid_raises(self):
        """Invalid method name should raise ValueError."""
        data = _generate_monthly_data(200)
        engine = BootstrapEngine(replications=5, base_seed=42)

        with pytest.raises(ValueError):
            engine.run_method("INVALID", data)

    def test_with_blocks(self):
        """Engine should pass blocks to Regime sampler."""
        data = _generate_monthly_data(200)
        blocks = _create_test_blocks(200)
        engine = BootstrapEngine(replications=3, base_seed=42)

        results = engine.run_all(data, blocks=blocks)

        # Just verify it runs without error
        assert "Regime" in results

    def test_deterministic_reproduction(self):
        """Same base_seed should produce reproducible results."""
        data = _generate_monthly_data(200)
        engine1 = BootstrapEngine(replications=5, base_seed=42)
        engine2 = BootstrapEngine(replications=5, base_seed=42)

        results1 = engine1.run_all(data)
        results2 = engine2.run_all(data)

        for method in METHOD_REGISTRY:
            for r1, r2 in zip(results1[method], results2[method]):
                pd.testing.assert_frame_equal(r1, r2)