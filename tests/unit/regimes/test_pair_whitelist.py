"""Test pair-specific transition whitelists (v14 fix for Reviewer §3)."""
import pytest

from mc_regime.regimes.whitelist import (
    allowed_transitions_for_pair,
    DEFAULT_EURUSD_WHITELIST,
    DEFAULT_GBPUSD_WHITELIST,
    DEFAULT_USDJPY_WHITELIST,
)


def test_eurusd_whitelist_is_eur_fed_specific():
    """The EURUSD whitelist encodes EUR/Fed policy transitions."""
    assert (1, 2) in DEFAULT_EURUSD_WHITELIST
    assert (5, 6) in DEFAULT_EURUSD_WHITELIST


def test_gbpusd_whitelist_reflects_boe_fed():
    """The GBPUSD whitelist encodes BoE/Fed transitions."""
    transitions = allowed_transitions_for_pair("GBPUSD")
    assert (1, 2) in transitions
    assert (5, 6) in transitions


def test_usdjpy_whitelist_reflects_boj_fed():
    """The USDJPY whitelist encodes BoJ/Fed transitions."""
    transitions = allowed_transitions_for_pair("USDJPY")
    assert (1, 2) in transitions
    assert (5, 6) in transitions


def test_unsupported_pair_raises():
    with pytest.raises(ValueError, match="Unsupported pair"):
        allowed_transitions_for_pair("AUDUSD")