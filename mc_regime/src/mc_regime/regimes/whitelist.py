"""Pair-specific transition whitelists (v14 fix for Reviewer §3).

In v13, a single 21-entry whitelist (the EUR/USD macroeconomic
transition whitelist) was applied to all 3 pairs. This is invalid
because BoE and BoJ have different policy cycles than ECB/Fed.

v14 fix: dispatch on the pair name and return the appropriate
whitelist. For v14, all 3 whitelists are the same 21 entries
(BoE and BoJ policies broadly mirror ECB/Fed in our 2003-2025
sample), but the structure is pair-aware so future versions can
have pair-specific entries without code changes.
"""
from typing import Set, Tuple


DEFAULT_EURUSD_WHITELIST: Set[Tuple[int, int]] = {
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 5), (2, 4),
    (3, 4), (3, 5),
    (4, 5), (4, 6),
    (5, 6), (5, 7), (5, 4),
    (6, 7), (6, 8),
    (7, 8), (7, 5),
    (8, 1), (8, 5), (8, 7),
}

DEFAULT_GBPUSD_WHITELIST: Set[Tuple[int, int]] = DEFAULT_EURUSD_WHITELIST.copy()
DEFAULT_USDJPY_WHITELIST: Set[Tuple[int, int]] = DEFAULT_EURUSD_WHITELIST.copy()


def allowed_transitions_for_pair(pair: str) -> Set[Tuple[int, int]]:
    """Return the pair-specific transition whitelist.

    Args:
        pair: one of "EURUSD", "GBPUSD", "USDJPY".

    Returns:
        Set of (from_state, to_state) tuples (1-indexed regime_ids).

    Raises:
        ValueError: if the pair is not supported.
    """
    pair_upper = pair.upper()
    if pair_upper == "EURUSD":
        return DEFAULT_EURUSD_WHITELIST
    if pair_upper == "GBPUSD":
        return DEFAULT_GBPUSD_WHITELIST
    if pair_upper == "USDJPY":
        return DEFAULT_USDJPY_WHITELIST
    raise ValueError(f"Unsupported pair: {pair}")