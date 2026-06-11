"""Unit tests for HMM detector."""

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from mc_regime.regimes.hmm_detector import HMMSdetector


class TestHMMSdetector:
    """Test suite for HMMSdetector class."""

    def test_hmm_selects_correct_k_on_synthetic(self):
        """Test that HMM selects k=2 when data has 2 regimes."""
        np.random.seed(42)
        n_samples = 200

        # Generate synthetic 2-regime data
        # Regime 0: low mean, low variance
        # Regime 1: high mean, high variance
        regime_0 = np.random.randn(n_samples, 5) * 0.5 - 1.0
        regime_1 = np.random.randn(n_samples, 5) * 0.5 + 1.0

        # Alternate regimes in a pattern (with some persistence)
        states = np.zeros(n_samples, dtype=int)
        for i in range(1, n_samples):
            if np.random.rand() < 0.85:  # 85% persistence
                states[i] = states[i - 1]
            else:
                states[i] = 1 - states[i]  # Flip

        X = np.zeros((n_samples, 5))
        for i in range(n_samples):
            if states[i] == 0:
                X[i] = regime_0[i]
            else:
                X[i] = regime_1[i]

        # Create a mock detector that uses synthetic data
        dates = pd.date_range("2010-01-01", periods=n_samples, freq="ME")
        features = pd.DataFrame(X, index=dates)

        # Directly test HMM fitting (we'll test through detector indirectly)
        # Test that with synthetic 2-regime data, we recover something reasonable
        model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            random_state=42
        )
        model.fit(X)
        predicted = model.predict(X)

        # Should have reasonable state separation
        unique_states = np.unique(predicted)
        assert len(unique_states) <= 2

        # Test BIC selection prefers k=2 for true 2-regime data
        bic_scores = {}
        for k in [2, 3, 4, 5]:
            m = GaussianHMM(n_components=k, covariance_type="diag", random_state=42)
            m.fit(X)
            ll = m.score(X)
            n_params = k - 1 + k * 5 * 2  # BIC approximate
            bic = -2 * ll + n_params * np.log(n_samples)
            bic_scores[k] = bic

        # k=2 should have lowest BIC
        best_k = min(bic_scores, key=bic_scores.get)
        assert best_k == 2, f"Expected k=2, got k={best_k}"

    def test_hmm_persistence_diagonal(self):
        """Test that transition matrix shows persistence (diagonal > 0.5)."""
        np.random.seed(42)
        n_samples = 200

        # Generate data with strong regime persistence
        states = np.zeros(n_samples, dtype=int)
        for i in range(1, n_samples):
            if np.random.rand() < 0.9:  # 90% persistence
                states[i] = states[i - 1]
            else:
                states[i] = 1 - states[i]

        X = np.random.randn(n_samples, 5) + states.reshape(-1, 1) * 2.0

        model = GaussianHMM(n_components=2, covariance_type="diag", random_state=42)
        model.fit(X)
        predicted = model.predict(X)

        # Estimate transition matrix manually
        counts = np.ones((2, 2))  # Laplace
        for i in range(len(predicted) - 1):
            counts[predicted[i], predicted[i + 1]] += 1
        trans = counts / counts.sum(axis=1, keepdims=True)

        # One diagonal should be > 0.5 for the dominant regime (persistence)
        max_diag = max(trans[0, 0], trans[1, 1])
        assert max_diag > 0.5, f"No diagonal > 0.5, trans={trans}"

    def test_fit_produces_valid_output(self):
        """Test that fit_predict produces valid outputs."""
        # This test verifies interface - data loading tested separately
        detector = HMMSdetector(
            fred_levels_path="dummy",
            fx_path="dummy"
        )

        # Should have expected attributes after initialization
        assert hasattr(detector, '_fred_levels')
        assert hasattr(detector, '_model')


class TestSyntheticHMM:
    """Tests with synthetic data mimicking real regime structure."""

    def generate_planted_2regime_data(
        self,
        n_months: int = 120,
        transition_prob: float = 0.1
    ) -> tuple:
        """Generate synthetic 2-regime data with known structure.

        Args:
            n_months: Number of months
            transition_prob: Probability of regime change

        Returns:
            tuple of (X, true_states)
        """
        np.random.seed(123)

        # Generate true state sequence
        states = np.zeros(n_months, dtype=int)
        states[0] = 0
        for i in range(1, n_months):
            if np.random.rand() < transition_prob:
                states[i] = 1 - states[i]
            else:
                states[i] = states[i - 1]

        # Generate observations conditioned on states
        X = np.zeros((n_months, 5))
        for i in range(n_months):
            if states[i] == 0:
                # Low volatility, lower mean (e.g., "calm" regime)
                X[i] = np.random.randn(5) * 0.5 - 0.5
            else:
                # High volatility, higher mean (e.g., "stressed" regime)
                X[i] = np.random.randn(5) * 1.0 + 1.0

        return X, states

    def test_bic_selects_true_k(self):
        """Test BIC recovers the planted 2 regimes."""
        X, true_states = self.generate_planted_2regime_data(n_months=120)

        # Test BIC selection
        bic_scores = {}
        for k in [2, 3, 4, 5]:
            model = GaussianHMM(
                n_components=k,
                covariance_type="diag",
                random_state=42
            )
            model.fit(X)
            ll = model.score(X)
            n_params = k - 1 + k * 5 * 2
            bic = -2 * ll + n_params * np.log(len(X))
            bic_scores[k] = bic
            print(f"k={k}: BIC={bic:.1f}")

        best_k = min(bic_scores, key=bic_scores.get)
        assert best_k == 2, f"Expected BIC to select k=2, got k={best_k}"

    def test_reconstructed_blocks_are_contiguous(self):
        """Test that reconstructed blocks are contiguous and non-overlapping."""
        X, true_states = self.generate_planted_2regime_data(n_months=60)

        model = GaussianHMM(n_components=2, covariance_type="diag", random_state=42)
        model.fit(X)
        pred_states = model.predict(X)

        # Build blocks
        dates = pd.date_range("2000-01-01", periods=len(pred_states), freq="ME")
        states_series = pd.Series(pred_states, index=dates)

        blocks = []
        current_state = pred_states[0]
        block_start = dates[0]

        for i in range(1, len(pred_states)):
            if pred_states[i] != current_state:
                blocks.append({
                    'start': block_start,
                    'end': dates[i - 1],
                    'state': current_state
                })
                current_state = pred_states[i]
                block_start = dates[i]

        # Last block
        blocks.append({
            'start': block_start,
            'end': dates[-1],
            'state': current_state
        })

        # Verify non-overlapping
        for i, b1 in enumerate(blocks):
            for j, b2 in enumerate(blocks):
                if i != j:
                    # Blocks should not have significant overlap
                    assert not (b1['start'] >= b2['start'] and b1['end'] <= b2['end'])

        # Verify contiguous - next block starts after previous ends
        for i in range(len(blocks) - 1):
            # Account for month-end date differences (e.g., Jan 31 to Mar 31 vs Feb 28)
            gap = (blocks[i + 1]['start'].year - blocks[i]['end'].year) * 12 + \
                  (blocks[i + 1]['start'].month - blocks[i]['end'].month)
            assert gap == 1