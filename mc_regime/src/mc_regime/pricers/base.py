"""Base classes for FX fair-value pricers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Series


class BasePricer(ABC):
    """Abstract base class for all pricers.

    Subclasses must implement fit() and fair_value() methods.
    """

    @abstractmethod
    def fit(self, data: "DataFrame") -> "BasePricer":
        """Estimate model parameters.

        Args:
            data: DataFrame containing dependent and independent variables.

        Returns:
            self for method chaining.
        """
        ...

    @abstractmethod
    def fair_value(self, data: "DataFrame") -> "Series":
        """Compute fitted fair values.

        Args:
            data: DataFrame with feature columns.

        Returns:
            Series of fitted fair values aligned with input index.
        """
        ...


class CointegratedPricerMixin:
    """Mixin for cointegrated models.

    Provides cointegration testing and residual-based diagnostics.
    """

    coint_pvalue_: float | None = None
    residuals_: "Series | None" = None

    def test_cointegration(self, y: "Series", x: "DataFrame") -> float:
        """Test for cointegration between y and x.

        Uses Engle-Granger two-step method via statsmodels.coint.

        Args:
            y: Target series (level data).
            x: Feature DataFrame (level data).

        Returns:
            Cointegration p-value.
        """
        from statsmodels.tsa.stattools import coint

        # Align indices
        common_idx = y.dropna().index.intersection(x.dropna().index)
        y_align = y.loc[common_idx]
        x_align = x.loc[common_idx]

        if len(y_align) < 20:
            return 1.0  # Too few observations

        # Test cointegration
        try:
            score, pvalue, _ = coint(y_align.values, x_align.values)
            return pvalue
        except Exception:
            return 1.0  # Default to no cointegration