"""Moving Block Bootstrap sampler (Künsch 1989).

This sampler resamples contiguous blocks of observations while sliding
through the time series, preserving local autocorrelation structure.
"""

from typing import Optional

import numpy as np
import pandas as pd

from mc_regime.bootstrap.samplers.base import Sampler


class MovingBlockSampler:
    """Moving Block Bootstrap (Künsch 1989).

    Resamples blocks of consecutive observations. The block length
    is typically set to T^(1/3) for optimal bandwidth.

    Args:
        block_length: Fixed block length. If None, computed as T^(1/3).
    """

    def __init__(self, block_length: Optional[int] = None):
        self.block_length = block_length

    def _compute_block_length(self, T: int) -> int:
        """Compute block length as T^(1/3) if not provided."""
        if self.block_length is not None:
            return self.block_length
        # T^(1/3) with minimum of 1
        return max(1, int(round(T ** (1 / 3))))

    def sample(
        self,
        data: pd.DataFrame,
        n: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Resample `n` rows from data using moving blocks.

        Args:
            data: Original DataFrame to resample from.
            n: Number of rows in the resampled output.
            rng: Random number generator.

        Returns:
            Resampled DataFrame with `n` rows.
        """
        T = len(data)
        block_len = self._compute_block_length(T)

        # Number of blocks needed
        n_blocks = int(np.ceil(n / block_len))

        # Sample starting positions for each block
        max_start = T - block_len
        starts = rng.integers(0, max_start + 1, size=n_blocks)

        # Build resampled data by concatenating blocks
        resampled_parts = []
        for start in starts:
            end = start + block_len
            resampled_parts.append(data.iloc[start:end])

        # Concatenate and truncate to desired length
        result = pd.concat(resampled_parts, ignore_index=True)
        return result.iloc[:n].reset_index(drop=True)