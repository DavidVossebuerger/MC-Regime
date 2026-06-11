"""Stationary Bootstrap sampler (Politis & Romano 1994).

This sampler resamples blocks with randomly varying geometrically-
distributed lengths, ensuring stationarity of the bootstrap distribution.
"""

from typing import Optional

import numpy as np
import pandas as pd

from mc_regime.bootstrap.samplers.base import Sampler


class StationarySampler:
    """Stationary Bootstrap (Politis & Romano 1994).

    Resamples blocks with random geometrically-distributed lengths.
    This produces a stationary bootstrap distribution.

    Args:
        length_param: Mean block length parameter.
            If None, defaults to the optimal T^(1/3) approximation.
    """

    def __init__(self, length_param: Optional[float] = None):
        self.length_param = length_param

    def _draw_block_lengths(
        self,
        n_blocks: int,
        T: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw random geometric block lengths.

        Args:
            n_blocks: Number of blocks to draw.
            T: Length of original time series.
            rng: Random number generator.

        Returns:
            Array of block lengths.
        """
        # Parameter p = 1/mean_length
        if self.length_param is not None:
            mean_length = self.length_param
        else:
            # Optimal T^(1/3) approximation
            mean_length = max(1, T ** (1 / 3))

        p = 1.0 / mean_length

        # Draw geometrically distributed lengths
        lengths = rng.geometric(p, size=n_blocks)

        # Ensure blocks don't exceed available data
        lengths = np.minimum(lengths, T)

        return lengths

    def sample(
        self,
        data: pd.DataFrame,
        n: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Resample `n` rows from data using stationary bootstrap.

        Args:
            data: Original DataFrame to resample from.
            n: Number of rows in the resampled output.
            rng: Random number generator.

        Returns:
            Resampled DataFrame with `n` rows.
        """
        T = len(data)

        # Generate blocks until we have at least n samples
        resampled_parts = []
        current_len = 0

        while current_len < n:
            # Draw random block length
            block_len = self._draw_block_lengths(1, T, rng)[0]

            # Draw random start position
            start = rng.integers(0, T)

            # Handle wrap-around for blocks near end
            if start + block_len > T:
                # Wrap around to beginning
                wrap_len = T - start
                remaining = block_len - wrap_len
                block_data = pd.concat([
                    data.iloc[start:],
                    data.iloc[:remaining],
                ], ignore_index=True)
            else:
                block_data = data.iloc[start:start + block_len]

            resampled_parts.append(block_data)
            current_len += block_len

        # Concatenate and truncate to desired length
        result = pd.concat(resampled_parts, ignore_index=True)
        return result.iloc[:n].reset_index(drop=True)