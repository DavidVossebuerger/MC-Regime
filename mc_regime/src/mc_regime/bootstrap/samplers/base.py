"""Base sampler protocol for bootstrap methods."""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Sampler(Protocol):
    """Protocol defining the bootstrap sampler interface."""

    def sample(
        self,
        data: pd.DataFrame,
        n: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Return resampled DataFrame with `n` rows.

        Args:
            data: Original data to resample from.
            n: Number of rows in the resampled output.
            rng: Random number generator for reproducibility.

        Returns:
            Resampled DataFrame with `n` rows.
        """
        ...