"""Block provider implementation using HMM-detected regimes."""

import numpy as np
import pandas as pd
from typing import List, Optional

from mc_regime.regimes.base import Block


class HMMBlockProvider:
    """Provides regime blocks from HMM-detected states.

    Takes predicted HMM states and constructs contiguous blocks
    where the regime id is constant.

    Attributes:
        states_series: Series of regime state indices indexed by date
        n_states: Number of unique states
        transition_matrix: State transition probabilities
    """

    def __init__(self, states_series: pd.Series):
        """Initialize the block provider.

        Args:
            states_series: Series with regime state indices, indexed by dates
        """
        self._states_series = states_series.sort_index()
        if len(states_series) == 0:
            self._n_states = 0
        else:
            self._n_states = int(states_series.max()) + 1
        self._blocks: List[Block] = []
        self._transition_matrix: Optional[pd.DataFrame] = None

        self._build_blocks()

    def _build_blocks(self) -> None:
        """Build contiguous blocks from state sequence."""
        states = self._states_series.values
        dates = self._states_series.index

        if len(states) == 0:
            return

        current_state = states[0]
        block_start = dates[0]

        for i in range(1, len(states)):
            if states[i] != current_state:
                # Close current block and start new one
                block_end = dates[i - 1]
                self._blocks.append(Block(
                    start=block_start,
                    end=block_end,
                    regime_id=int(current_state)
                ))
                current_state = states[i]
                block_start = dates[i]

        # Don't forget the last block
        block_end = dates[-1]
        self._blocks.append(Block(
            start=block_start,
            end=block_end,
            regime_id=int(current_state)
        ))

    def get_blocks(
        self,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None
    ) -> List[Block]:
        """Get regime blocks in the given date range.

        Args:
            start_date: Start of date range (None = earliest)
            end_date: End of date range (None = latest)

        Returns:
            List of Block objects
        """
        if start_date is None:
            start_date = self._states_series.index.min()
        if end_date is None:
            end_date = self._states_series.index.max()

        return [
            b for b in self._blocks
            if b.start <= end_date and b.end >= start_date
        ]

    @property
    def n_states(self) -> int:
        """Number of regime states."""
        return self._n_states

    @property
    def transition_matrix(self) -> pd.DataFrame:
        """State transition probability matrix."""
        return self._estimate_transition_matrix()

    def _estimate_transition_matrix(self) -> pd.DataFrame:
        """Estimate transition matrix with Laplace smoothing.

        Returns:
            DataFrame with transition probabilities
        """
        states = self._states_series.values
        n_states = self._n_states

        # Initialize with Laplace smoothing (+1)
        counts = np.ones((n_states, n_states))

        # Count transitions
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            counts[from_state, to_state] += 1

        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = counts / row_sums

        # Create DataFrame with state labels
        labels = [f"state_{i}" for i in range(n_states)]
        self._transition_matrix = pd.DataFrame(
            probs,
            index=labels,
            columns=labels
        )

        return self._transition_matrix


def create_block_provider(states_series: pd.Series) -> HMMBlockProvider:
    """Factory function to create an HMMBlockProvider."""
    return HMMBlockProvider(states_series)