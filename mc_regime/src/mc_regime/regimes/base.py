"""Base classes for regime blocks."""

from dataclasses import dataclass
from typing import Protocol, List
import pandas as pd


@dataclass
class Block:
    """Represents a contiguous regime block.

    Attributes:
        start: Start timestamp (month-end)
        end: End timestamp (month-end)
        regime_id: Integer identifier for the regime state
    """
    start: pd.Timestamp
    end: pd.Timestamp
    regime_id: int

    @property
    def length_months(self) -> int:
        """Return the length of the block in months."""
        return ((self.end.year - self.start.year) * 12 +
                (self.end.month - self.start.month) + 1)


class BlockProvider(Protocol):
    """Protocol for providing regime blocks.

    Implementations should provide methods to get blocks
    over a given date range.
    """
    def get_blocks(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Block]:
        """Get regime blocks in the given date range.

        Args:
            start_date: Start of the date range
            end_date: End of the date range

        Returns:
            List of Block objects
        """
        ...

    @property
    def n_states(self) -> int:
        """Number of regime states."""
        ...

    @property
    def transition_matrix(self) -> pd.DataFrame:
        """State transition probability matrix."""
        ...


# Alias for export in __init__.py
BlockProviderProto = BlockProvider