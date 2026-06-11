"""Structure preservation test for regime-preserving bootstrap.

This package implements the 4-level structure preservation test:

- Level 1: Correlation bias (mean(boot) - orig)  [contemporaneous]
- Level 2: Wasserstein-1 distance from point mass  [contemporaneous]
- Level 3: Network structure (Frobenius + spectral distances)
            * L3_fro: contemporaneous adjacency
            * L3_spec: spectral radius — captures temporal dynamics
- Level 4: Temporal / path-dependent metrics
            * lag-1 autocorrelations
            * variance ratios
            * max-drawdown distribution
            * lagged cross-correlations (FVG → FX)

Per Spec §3.4 and BUG-3-fix discussion: contemporaneous levels (L1, L2,
L3_fro) are trivially preserved by IID; the genuinely interesting test
is whether Markov-chain-simulated regime paths preserve *temporal* structure.
"""

from mc_regime.structure.arrangement import (
    arrangement_distance,
    compute_arrangement_distances,
)
from mc_regime.structure.composite import (
    compute_composite_loss,
    structure_preservation_report,
)
from mc_regime.structure.correlations import (
    DEFAULT_CORRELATION_PAIRS,
    compute_correlations,
    compute_correlation_bias,
    compute_l1_bias,
)
from mc_regime.structure.network import (
    build_correlation_graph,
    compute_l3_network,
    compute_network_distance,
)
from mc_regime.structure.temporal import (
    lag1_autocorr,
    lagk_cross_corr,
    max_drawdown,
    mean_temporal_loss,
    temporal_distance,
    temporal_metrics,
    variance_ratio,
)
from mc_regime.structure.wasserstein import (
    compute_l2_wasserstein,
    compute_wasserstein_distances,
)

__all__ = [
    # Arrangement
    "arrangement_distance",
    "compute_arrangement_distances",
    # Composite + reports
    "compute_composite_loss",
    "structure_preservation_report",
    # Contemporaneous
    "DEFAULT_CORRELATION_PAIRS",
    "compute_correlations",
    "compute_correlation_bias",
    "compute_l1_bias",
    "compute_wasserstein_distances",
    "compute_l2_wasserstein",
    "build_correlation_graph",
    "compute_network_distance",
    "compute_l3_network",
    # Temporal / path-dependent
    "lag1_autocorr",
    "lagk_cross_corr",
    "variance_ratio",
    "max_drawdown",
    "temporal_metrics",
    "temporal_distance",
    "mean_temporal_loss",
]
