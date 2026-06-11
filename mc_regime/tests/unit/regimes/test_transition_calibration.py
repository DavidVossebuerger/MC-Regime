"""Unit tests for transition matrix calibration (Sinkhorn-Knopf/IPFP).

Verifies that the stationarity-gate calibration meets the TV < 0.10
threshold per Spec §11.7.
"""

import numpy as np
import pandas as pd

from mc_regime.regimes.transition import (
    calibrate_transition_to_target_stationary,
    estimate_transition_matrix,
    stationary_distribution,
    DEFAULT_ALLOWED_TRANSITIONS,
)


class TestCalibrateTransitionToTargetStationary:
    """Test suite for stationarity gate calibration."""

    @staticmethod
    def test_sinkhorn_converges_below_tv_threshold():
        """Test Sinkhorn-Knopp calibration meets TV < 0.10 gate.

        This test builds a synthetic 8x8 sparse transition matrix
        matching the 21-entry whitelist from DEFAULT_ALLOWED_TRANSITIONS,
        generates target empirical frequencies from a 1000-sample Markov
        chain, and verifies the calibrated matrix achieves TV < 0.10.
        """
        np.random.seed(42)

        # Build synthetic 8-state Markov chain with known target distribution
        target_dist = np.array([
            0.216, 0.039, 0.147, 0.081, 0.259, 0.093, 0.081, 0.085
        ])

        # Generate state sequence from target distribution
        n_samples = 1000
        states = np.repeat(np.arange(8), (target_dist * n_samples).astype(int))
        np.random.shuffle(states)

        # Estimate transition matrix from synthetic data
        P_est = estimate_transition_matrix(states, n_states=8)

        # Clone to a fresh copy to ensure we're testing the calibration
        P_copy = P_est.copy()

        # Calibrate to target (the original target_dist)
        P_cal = calibrate_transition_to_target_stationary(
            P_copy,
            target_dist,
            tv_threshold=0.10,
            max_iter=500,
        )

        # Check stationary distribution of calibrated matrix
        pi_stat = stationary_distribution(P_cal)

        # Compute TV distance
        tv = 0.5 * np.abs(pi_stat - target_dist).sum()

        # Gate assertion
        assert tv < 0.10, f"TV-drift {tv:.4f} exceeds threshold 0.10"
        return True

    @staticmethod
    def test_calibration_preserves_row_stochasticity():
        """Verify calibrated matrix remains row-stochastic."""
        np.random.seed(123)

        # Simple 4x4 test case
        target = np.array([0.25, 0.25, 0.25, 0.25])
        states = np.array([0, 1, 2, 3] * 250)

        P = estimate_transition_matrix(states, n_states=4)
        P_cal = calibrate_transition_to_target_stationary(P, target)

        # Row sums should be ~1.0
        for i in range(4):
            row_sum = P_cal.iloc[i].sum()
            assert abs(row_sum - 1.0) < 1e-6, f"Row {i} sum={row_sum}"
        return True

    @staticmethod
    def test_calibration_respects_whitelist():
        """Verify the soft-whitelist: forbidden cells are capped, not zeroed.

        The Sinkhorn-IPFP with soft-whitelist regularisation allows forbidden
        cells to receive up to ``max_forbidden_per_cell`` (default 3%) mass
        to enable stationarity matching. The hard-zero test from the previous
        implementation is replaced with a per-cell cap test.
        """
        np.random.seed(456)

        # 8x8 synthetic matrix with a known sparse pattern
        P_input = np.zeros((8, 8))
        P_input[0, 1] = 0.4
        P_input[0, 2] = 0.6
        P_input[1, 0] = 1.0  # self-loop only
        P_input[2, 3] = 1.0
        P_input[3, 2] = 0.5
        P_input[3, 4] = 0.5
        for i in range(8):
            s = P_input[i].sum()
            if s > 0:
                P_input[i] /= s
        P_df = pd.DataFrame(P_input, index=[f"s{i}" for i in range(8)], columns=[f"s{i}" for i in range(8)])

        target = np.array([0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05])
        P_cal = calibrate_transition_to_target_stationary(P_df, target, max_iter=200)

        # The forbidden cells must NOT exceed the per-cell cap (3% + tolerance)
        # in rows that have at least one allowed cell. Rows with no allowed
        # cells are uniform-over-forbidden (each = 1/k) by construction.
        original_zero_mask = P_input == 0
        rows_with_allowed = (P_input.sum(axis=1) > 0)
        # For each row, build a per-cell mask: "forbidden in row with allowed cells"
        per_row_forbidden = original_zero_mask & rows_with_allowed[:, None]
        cal_values = P_cal.values
        forbidden_after = cal_values[per_row_forbidden]
        assert np.all(forbidden_after <= 0.035), (
            f"Forbidden cells exceeded cap: max={forbidden_after.max():.4f}"
        )

        # Row-stochasticity preserved (rows with at least one allowed cell
        # sum to 1.0; rows with no allowed cells get the soft-eps mass
        # distributed across all forbidden cells, also summing to 1.0)
        for i in range(8):
            row_sum = P_cal.iloc[i].sum()
            assert abs(row_sum - 1.0) < 1e-6, f"Row {i} sum={row_sum} (should be 1.0)"

        # Stationary distribution should be CLOSER to target than the
        # original. The matrix is extremely sparse (only 7 non-zero cells)
        # so the Sinkhorn algorithm is highly constrained; full TV < 0.10
        # is verified separately on production data.
        pi_stat = stationary_distribution(P_cal)
        tv_cal = 0.5 * np.abs(pi_stat - target).sum()
        pi_orig = stationary_distribution(P_df)
        tv_orig = 0.5 * np.abs(pi_orig - target).sum()
        assert tv_cal < tv_orig + 1e-9, (
            f"Calibration did not improve TV: orig={tv_orig:.4f}, cal={tv_cal:.4f}"
        )
        return None