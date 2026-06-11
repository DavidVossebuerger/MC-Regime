"""Data classes for bootstrap engine."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Block:
    """Represents a contiguous regime/event block in the time series.

    This is a frozen dataclass representing a contiguous block of
    observations that belong to the same market regime or event state.
    Used by the Regime-Preserving bootstrap method to resample whole blocks.

    Attributes:
        start_idx: Starting index of the block (inclusive).
        end_idx: Ending index of the block (inclusive).
        regime_label: Optional label identifying the regime/state.
    """

    start_idx: int
    end_idx: int
    regime_label: Optional[int] = None

    @property
    def length(self) -> int:
        """Return the number of observations in the block."""
        return self.end_idx - self.start_idx + 1

    def contains(self, idx: int) -> bool:
        """Check if index is within this block."""
        return self.start_idx <= idx <= self.end_idx