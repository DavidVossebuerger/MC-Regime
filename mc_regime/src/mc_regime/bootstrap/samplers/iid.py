"""IID bootstrap sampler (Efron 1979).

This sampler performs i.i.d. resampling by randomly selecting individual
observations with replacement from the original dataset.
"""

import numpy as np
import pandas as pd

from mc_regime.bootstrap.samplers.base import Sampler


class IIDSampler:
    """IID bootstrap (Efron 1979).

    Resamples individual rows with replacement. This breaks any temporal
    dependence in the data, making it suitable for i.i.d. assumptions.
    """

    def sample(
        self,
        data: pd.DataFrame,
        n: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Resample `n` rows from data using IID bootstrap.

        Args:
            data: Original DataFrame to resample from.
            n: Number of rows in the resampled output.
            rng: Random number generator.

        Returns:
            Resampled DataFrame with `n` rows.
        """
        # Get indices to resample with replacement
        original_indices = rng.integers(0, len(data), size=n)
        return data.iloc[original_indices].reset_index(drop=True)