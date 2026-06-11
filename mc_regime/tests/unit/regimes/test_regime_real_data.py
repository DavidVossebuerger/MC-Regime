"""Integration test with real FRED and FX data."""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, '/root/research/MC-Regime/mc_regime/src')

from mc_regime.regimes.hmm_detector import HMMSdetector, create_detector
from mc_regime.regimes.hmm_blocks import HMMBlockProvider


class TestRegimeRealData:
    """Integration tests using real FRED + FX data."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Load real data and build detector."""
        fred_path = "/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_levels.csv"
        fx_path = "/root/research/MC-Regime/mc_regime/outputs/data/EURUSD_1h.parquet"

        detector = HMMSdetector(
            fred_levels_path=fred_path,
            fx_path=fx_path,
            fred_yoy_path=None  # Compute YoY from levels data
        )

        # Fit and get states
        n_states, states = detector.fit_predict()

        return detector, n_states, states

    def test_features_built(self, detector):
        """Verify feature construction."""
        det, _, _ = detector
        features = det.features

        print("\n=== Features ===")
        print(f"Shape: {features.shape}")
        print(f"Date range: {features.index.min()} to {features.index.max()}")
        print(f"Columns: {features.columns.tolist()}")
        print(features.head())

        assert features.shape[1] == 5
        assert not features.isnull().any().any()

    def test_bic_selection(self, detector):
        """Test BIC-based k selection with real data."""
        det, n_states, _ = detector

        print("\n=== BIC Selection ===")
        print(f"Selected k={n_states}")

        assert n_states in [2, 3, 4, 5]

    def test_state_sequence(self, detector):
        """Verify state sequence."""
        det, n_states, states = detector

        print("\n=== States ===")
        print(f"Unique states: {np.unique(states)}")
        print(f"Sample states:\n{states[:10]}")

        assert len(states) == len(det.features)
        assert states.min() >= 0
        assert states.max() < n_states

    def test_block_construction(self, detector):
        """Test block construction on real data."""
        det, n_states, states = detector

        states_series = det.get_states_series()
        provider = HMMBlockProvider(states_series)

        blocks = provider.get_blocks()

        print("\n=== Blocks ===")
        print(f"Number of blocks: {len(blocks)}")
        print(f"n_states: {provider.n_states}")

        # Print sample blocks
        for i, b in enumerate(blocks[:10]):
            print(f"Block {i}: {b.start.date()} to {b.end.date()}, regime={b.regime_id}, length={b.length_months}mo")

        assert len(blocks) > 0

        # Blocks should be sorted by date
        for i in range(len(blocks) - 1):
            assert blocks[i].start < blocks[i + 1].start
            assert blocks[i].end < blocks[i + 1].start or blocks[i].end + pd.DateOffset(months=1) >= blocks[i + 1].start

    def test_transition_matrix(self, detector):
        """Test transition matrix estimation."""
        det, n_states, states = detector

        states_series = det.get_states_series()
        provider = HMMBlockProvider(states_series)

        tm = provider.transition_matrix

        print("\n=== Transition Matrix ===")
        print(tm)

        # Check validity
        assert tm.shape == (n_states, n_states)
        assert np.allclose(tm.values.sum(axis=1), 1.0)

        # Diagonal should generally be high (> 0.3) for persistence
        for i in range(n_states):
            assert tm.iloc[i, i] > 0.3, f"Diagonal entry [{i},{i}] = {tm.iloc[i,i]:.3f} is too low"

    def test_full_pipeline_summary(self, detector):
        """Full pipeline summary for reporting."""
        det, n_states, states = detector

        states_series = det.get_states_series()
        provider = HMMBlockProvider(states_series)
        blocks = provider.get_blocks()
        tm = provider.transition_matrix

        print("\n" + "="*50)
        print("REAL DATA INTEGRATION TEST SUMMARY")
        print("="*50)
        print(f"Features: {det.features.shape[1]} columns")
        print(f"Date range: {det.features.index.min().date()} to {det.features.index.max().date()}")
        print(f"BIC-selected k: {n_states}")
        print(f"Total blocks: {len(blocks)}")
        print(f"Transition matrix:")
        print(tm.to_string())
        print("="*50)

        # Basic assertions
        assert n_states >= 2, "Should detect at least 2 regimes"
        assert len(blocks) >= n_states, "Should have at least as many blocks as states"


# Entry point for direct execution
if __name__ == "__main__":
    print("Running real data integration test...")

    fred_path = "/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_levels.csv"
    fx_path = "/root/research/MC-Regime/mc_regime/outputs/data/EURUSD_1h.parquet"
    yoy_path = "/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_yoy.csv"

    detector = HMMSdetector(
        fred_levels_path=fred_path,
        fx_path=fx_path,
        fred_yoy_path=yoy_path
    )

    n_states, states = detector.fit_predict()

    # Build blocks
    states_series = detector.get_states_series()
    provider = HMMBlockProvider(states_series)
    blocks = provider.get_blocks()
    tm = provider.transition_matrix

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"BIC-selected k: {n_states}")
    print(f"Number of blocks: {len(blocks)}")
    print(f"\nTransition Matrix:")
    print(tm)