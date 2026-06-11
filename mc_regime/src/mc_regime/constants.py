"""Shared constants for the MC-Regime package."""

from enum import Enum


class BootMethod(str, Enum):
    """Bootstrap method names."""
    IID = "IID"
    MBB = "MBB"
    SB = "SB"
    REGIME_UNIFORM = "Regime-Uniform"
    REGIME_MARKOV = "Regime-Markov"


class LossLevel(str, Enum):
    """Structure-preservation loss levels."""
    L0 = "L0"  # Performance gap (Sharpe)
    L1 = "L1"  # Contemporaneous correlation
    L2 = "L2"  # Wasserstein of contemporaneous
    L3_FRO = "L3_fro"  # Network Frobenius
    L3_SPEC = "L3_spec"  # Network spectral
    L4_TEMPORAL = "L4_temporal"  # Path-dependent