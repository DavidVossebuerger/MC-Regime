"""Gap calculation utilities for fair-value models.

The gap represents the deviation of actual FX from the model's fair value,
often interpreted as undervaluation (+) or overvaluation (-) in the currency.
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_gap(
    actual: pd.Series,
    fitted: pd.Series,
    method: str = "log_diff",
) -> pd.Series:
    """Compute fair-value gap between actual and fitted FX values.

    Args:
        actual: Actual FX values (level or log).
        fitted: Fitted fair values (level or log).
        method: Gap computation method:
            - 'log_diff': log(actual) - log(fitted) [default]
            - 'level_diff': actual - fitted (after aligning log scales)

    Returns:
        Series of gaps aligned with common index.

    Example:
        >>> actual_log = log(EURUSD_actual)
        >>> fitted_log = model.fair_value(data)
        >>> gap = compute_gap(actual_log, fitted_log)
    """
    # Determine if values are in log or level space
    # Heuristic: if values are typically > 10, assume level
    actual_mean = actual.mean() if len(actual) > 0 else 1
    use_level = actual_mean > 10

    if method == "log_diff":
        # Both should be in log space
        gap = actual - fitted
    elif method == "level_diff":
        # Convert to level first if in log space
        if use_level:
            act_level = actual
            fit_level = fitted
        else:
            act_level = np.exp(actual)
            fit_level = np.exp(fitted)

        gap = act_level - fit_level
    else:
        raise ValueError(f"Unknown method: {method}")

    # Drop NaNs
    gap = gap.dropna()

    return gap


def gap_statistics(gap: pd.Series) -> dict:
    """Compute summary statistics on gap series.

    Args:
        gap: Gap series from compute_gap.

    Returns:
        Dict with mean, std, min, max, percentiles (5, 25, 50, 75, 95).
    """
    return {
        "mean": float(gap.mean()),
        "std": float(gap.std()),
        "min": float(gap.min()),
        "max": float(gap.max()),
        "p5": float(gap.quantile(0.05)),
        "p25": float(gap.quantile(0.25)),
        "p50": float(gap.median()),
        "p75": float(gap.quantile(0.75)),
        "p95": float(gap.quantile(0.95)),
        "n_obs": int(len(gap)),
    }


def align_for_gap(
    data: pd.DataFrame,
    log_col: str = "log_FX",
    fitted_col: Optional[str] = None,
) -> tuple[pd.Series, pd.Series]:
    """Align actual and fitted series for gap calculation.

    Args:
        data: DataFrame with log_FX and fitted columns.
        log_col: Column name for log FX.
        fitted_col: Column name for fitted values. If None, looks for 'fitted'.

    Returns:
        Tuple of (actual_log, fitted_log) series.
    """
    actual = data[log_col].dropna()

    fit_col = fitted_col if fitted_col else "fitted"
    if fit_col in data.columns:
        fitted = data[fit_col].dropna()
    elif "fair_value" in data.columns:
        fitted = data["fair_value"].dropna()
    else:
        raise ValueError(f"No fitted values found. Expected column: {fitted_col}")

    # Align indices
    common_idx = actual.index.intersection(fitted.index)

    return actual.loc[common_idx], fitted.loc[common_idx]


class GapAnalyzer:
    """Analyzer for fair-value gap time series.

    Provides methods for assessing convergence/divergence and
    outlier detection.
    """

    def __init__(self, gap: pd.Series):
        """Initialize with gap series.

        Args:
            gap: Gap series (actual_log - fitted_log).
        """
        self.gap = gap
        self.stats_ = gap_statistics(gap)

    def is_extreme(self, threshold: float = 2.0) -> pd.Series:
        """Identify extreme gap observations.

        Args:
            threshold: Number of standard deviations for extremes.

        Returns:
            Boolean series indicating extreme observations.
        """
        mean = self.stats_["mean"]
        std = self.stats_["std"]

        if std < 1e-9:
            return pd.Series(False, index=self.gap.index)

        z_score = (self.gap - mean) / std
        return z_score.abs() > threshold

    def regime_indicators(
        self,
        low_threshold: float = -1.5,
        high_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """Generate regime indicators based on gap percentiles.

        Distinguishes undervaluation, fair, and overvaluation regimes.

        Args:
            low_threshold: Lower z-score threshold for undervaluation.
            high_threshold: Upper z-score threshold for overvaluation.

        Returns:
            DataFrame with regime indicators.
        """
        mean = self.stats_["mean"]
        std = self.stats_["std"]

        z = (self.gap - mean) / std

        return pd.DataFrame({
            "gap": self.gap,
            "z_score": z,
            "is_undervalued": z < low_threshold,
            "is_overvalued": z > high_threshold,
            "is_fair": (z >= low_threshold) & (z <= high_threshold),
        })

    def convergence_test(
        self,
        window: int = 60,
        eps: float = 0.01,
    ) -> dict:
        """Test for gap convergence toward zero.

        Uses linear regression on recent window to detect trend.

        Args:
            window: Lookback window for convergence test.
            eps: Convergence threshold (gap magnitude).

        Returns:
            Dict with test results.
        """
        if len(self.gap) < window:
            window = len(self.gap)

        recent = self.gap.tail(window)

        # Simple test: is mean close to zero?
        mean_near_zero = abs(recent.mean()) < eps

        # Regression test for trend
        from scipy import stats

        x = np.arange(len(recent))
        y = recent.values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        return {
            "converged": mean_near_zero,
            "slope": float(slope),
            "slope_pvalue": float(p_value),
            "r_squared": float(r_value ** 2),
            "window": window,
        }

    def summary(self) -> dict:
        """Return full gap summary."""
        return {
            "statistics": self.stats_,
            "recent_mean": float(self.gap.tail(60).mean()),
            "recent_std": float(self.gap.tail(60).std()),
            "extremes_share": float(self.is_extreme(threshold=2.0).mean()),
        }