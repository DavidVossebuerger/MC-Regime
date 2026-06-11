"""BEER (Behavioural Equilibrium Exchange Rate) pricer with FMOLS.

Implements the Clark-MacDonald 1998 reduced-form BEER model using
FMOLS-style estimation with HAC (Newey-West) standard errors.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from mc_regime.pricers.base import BasePricer, CointegratedPricerMixin


class BEERPricer(BasePricer, CointegratedPricerMixin):
    """BEER pricer using Fully Modified OLS (FMOLS) style estimation.

    Clark-MacDonald 1998 reduced form:
        ln(Q_t) = α + β1·TOT_t + β2·NFA_t + β3·(Y_t − Y_t*) + β4·(r_t − r_t*) + γ'·Z_t + ε_t

    Features (mapping to FRED data):
        - TOT: Terms of trade (computed from CPI ratio changes)
        - NFA: Net foreign assets proxy (GDP ratio differential)
        - ProdDiff: Productivity differential (GDP growth differential)
        - RateDiff: Real interest rate differential
        - FiscalBalance: Placeholder for fiscal balance (Z control)

    Usage:
        >>> pricer = BEERPricer()
        >>> pricer.fit(data)
        >>> gap = pricer.gap(data)
    """

    FEATURES = ["TOT", "NFA", "ProdDiff", "RateDiff", "FiscalBalance"]

    def __init__(
        self,
        maxlag: int = 12,
        hac_kernel: str = "bartlett",
    ):
        """Initialize BEER pricer.

        Args:
            maxlag: Maximum lag for HAC covariance (Newey-West).
            hac_kernel: Kernel type for HAC estimation.
        """
        self.maxlag = maxlag
        self.hac_kernel = hac_kernel

        # Fitted attributes
        self.coefficients_: Optional[pd.Series] = None
        self.alpha_: Optional[float] = None
        self.std_errors_: Optional[pd.Series] = None
        self.residuals_: Optional[pd.Series] = None
        self.fitted_: Optional[pd.Series] = None
        self.coint_pvalue_: Optional[float] = None

        # Model fitted flag
        self.is_fitted_ = False

    @staticmethod
    def build_features(
        fx_data: pd.DataFrame,
        fred_levels: pd.DataFrame,
        fred_yoy: pd.DataFrame | None = None,
        pair: str = "EURUSD",
    ) -> pd.DataFrame:
        """Build BEER features from raw FX and FRED data.

        Args:
            fx_data: DataFrame with column 'close' (nominal FX rate).
            fred_levels: FRED panel data (levels).
            fred_yoy: Not used (kept for compatibility).
            pair: Currency pair (e.g. 'EURUSD', 'GBPUSD', 'USDJPY'). Used to
                select the right currency columns for non-EUR pairs.

        Returns:
            DataFrame with features: log_FX, TOT, NFA, ProdDiff, RateDiff, FiscalBalance.
        """
        import warnings

        warnings.filterwarnings("ignore")

        # Map currency pair to (base, quote) currencies
        pair_upper = pair.upper()
        if pair_upper == "EURUSD":
            base_ccy, quote_ccy = "EUR", "USD"
        elif pair_upper == "GBPUSD":
            base_ccy, quote_ccy = "GBP", "USD"
        elif pair_upper == "USDJPY":
            # USDJPY convention: base is USD, quote is JPY (price = JPY per USD)
            base_ccy, quote_ccy = "USD", "JPY"
        else:
            raise ValueError(f"Unsupported pair: {pair}")

        base_cpi = f"{base_ccy}_CPI"
        quote_cpi = f"{quote_ccy}_CPI"
        base_gdp = f"{base_ccy}_GDP"
        quote_gdp = f"{quote_ccy}_GDP"
        # Policy rates have different column names in the FRED cache
        rate_col_map = {
            "EUR": "EUR_Policy Rate",
            "GBP": "GBP_Policy Rate",
            "JPY": "JPY_Policy Rate",
            "USD": "USD_Fed Funds",
        }
        base_rate = rate_col_map[base_ccy]
        quote_rate = rate_col_map[quote_ccy]

        # Prepare FX data
        fx_df = fx_data.copy()
        if "timestamp" in fx_df.columns:
            fx_df["date"] = pd.to_datetime(fx_df["timestamp"]).dt.tz_localize(None)
        elif "date" in fx_df.columns:
            fx_df["date"] = pd.to_datetime(fx_df["date"]).dt.tz_localize(None)
        else:
            fx_df["date"] = pd.to_datetime(fx_df.index).dt.tz_localize(None)
        fx_df = fx_df.set_index("date").sort_index()

        # Prepare FRED levels data
        fred_df = fred_levels.copy()
        fred_df["date"] = pd.to_datetime(fred_df["date"])
        fred_df = fred_df.set_index("date").sort_index()

        # Forward fill FRED to daily
        fred_daily = fred_df.resample("D").last().ffill()
        # Backward fill to handle initial NaN (e.g., for series that start
        # later than the FX data); then fill any remaining NaN with 0
        # (which is a safe neutral default for BEER features).
        fred_daily = fred_daily.bfill().fillna(0)

        # Join with FX data (inner join to keep only matching dates)
        combined = fx_df.join(fred_daily, how="inner")

        # Compute log(FX) - generic (no EUR/USD convention)
        combined["log_FX"] = np.log(combined["close"])

        # ============= Compute derived features =============

        # 1. Terms of trade: base_CPI / quote_CPI relative change
        if base_cpi in combined.columns and quote_cpi in combined.columns:
            cpi_ratio = combined[base_cpi] / combined[quote_cpi]
            combined["TOT"] = cpi_ratio.pct_change().fillna(0) * 100
        else:
            combined["TOT"] = 0.0

        # 2. NFA proxy: Relative GDP level
        if base_gdp in combined.columns and quote_gdp in combined.columns:
            gdp_ratio = combined[base_gdp].ffill() / combined[quote_gdp].ffill()
            combined["NFA"] = (gdp_ratio - 1).fillna(0) * 100
        else:
            combined["NFA"] = 0.0

        # 3. Productivity differential: GDP growth rate differential
        if base_gdp in combined.columns and quote_gdp in combined.columns:
            base_gdp_yoy = combined[base_gdp].pct_change(periods=12).fillna(0) * 100
            quote_gdp_yoy = combined[quote_gdp].pct_change(periods=12).fillna(0) * 100
            combined["ProdDiff"] = (base_gdp_yoy - quote_gdp_yoy).fillna(0)
        else:
            combined["ProdDiff"] = 0.0

        # 4. Real rate differential: Nominal - Inflation (per currency)
        if quote_cpi in combined.columns:
            quote_cpi_yoy = combined[quote_cpi].pct_change(periods=12).fillna(0) * 100
        else:
            quote_cpi_yoy = pd.Series(0, index=combined.index)

        if base_cpi in combined.columns:
            base_cpi_yoy = combined[base_cpi].pct_change(periods=12).fillna(0) * 100
        else:
            base_cpi_yoy = pd.Series(0, index=combined.index)

        base_nominal = combined.get(base_rate).ffill().fillna(0)
        quote_nominal = combined.get(quote_rate).ffill().fillna(0)

        base_real = base_nominal - base_cpi_yoy
        quote_real = quote_nominal - quote_cpi_yoy
        combined["RateDiff"] = (quote_real - base_real).fillna(0)

        # 5. Fiscal balance placeholder
        combined["FiscalBalance"] = 0.0

        # Final NaN/Inf safety: replace any remaining infinities or NaNs
        # (e.g. from ratio=0 or pct_change on zero bases) with 0
        feature_cols = ["TOT", "NFA", "ProdDiff", "RateDiff", "FiscalBalance"]
        for c in feature_cols:
            combined[c] = combined[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return combined[["log_FX"] + feature_cols]

        # Fiscal balance placeholder (Z control)
        combined["FiscalBalance"] = 0.0

        return combined[["log_FX", "TOT", "NFA", "ProdDiff", "RateDiff", "FiscalBalance"]]

    def fit(self, data: pd.DataFrame) -> "BEERPricer":
        """Fit BEER model using FMOLS-style estimation.

        Steps:
            1. Test for unit root in log(FX) and fundamentals.
            2. Test cointegration.
            3. Run OLS with HAC (Newey-West) covariance.

        Args:
            data: DataFrame with columns:
                - log_FX: log of nominal exchange rate
                - TOT, NFA, ProdDiff, RateDiff, FiscalBalance: feature columns

        Returns:
            self for chaining.
        """
        # Validate input
        required_cols = ["log_FX"] + self.FEATURES
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Drop NaN rows
        clean = data[required_cols].dropna()
        if len(clean) < 20:
            raise ValueError("Insufficient data for estimation (< 20 obs)")

        y = clean["log_FX"]
        X = clean[self.FEATURES]

        # Step 1: Test for unit root (ADF) on log(FX) and key fundamentals
        # Skip if too few observations
        self._adf_y = adfuller(y, autolag="AIC")[1] if len(y) >= 20 else 1.0

        # Step 2: Test cointegration
        self.coint_pvalue_ = self.test_cointegration(y, X)

        # Step 3: Run OLS with HAC covariance
        # Add constant
        X_const = sm.add_constant(X)

        # Fit OLS
        model = OLS(y, X_const)
        self._ols_result = model.fit()

        # HAC covariance (Newey-West)
        try:
            hac = model.fit(cov_type="HAC", cov_kwds={"maxlags": self.maxlag})
            use_hac = True
        except Exception:
            # Fallback to OLS if HAC fails
            hac = self._ols_result
            use_hac = False

        # Extract results
        if use_hac:
            self.coefficients_ = pd.Series(hac.params, index=X_const.columns)
            self.std_errors_ = pd.Series(hac.bse, index=X_const.columns)
        else:
            self.coefficients_ = pd.Series(self._ols_result.params, index=X_const.columns)
            self.std_errors_ = pd.Series(self._ols_result.bse, index=X_const.columns)

        self.alpha_ = float(self.coefficients_.iloc[0])
        self.residuals_ = hac.resid
        self.fitted_ = pd.Series(hac.fittedvalues, index=y.index)

        self.is_fitted_ = True

        return self

    def fair_value(self, data: pd.DataFrame) -> pd.Series:
        """Compute fitted fair value in log space.

        Args:
            data: DataFrame with feature columns.

        Returns:
            Series of fitted fair values (log scale) aligned with input index.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        required = self.FEATURES
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = data[required].dropna()
        X_const = sm.add_constant(X)

        fitted_vals = self.coefficients_.dot(X_const.T)

        result = pd.Series(fitted_vals, index=X.index)
        return result

    def gap(self, data: pd.DataFrame) -> pd.Series:
        """Compute fair-value gap.

        gap_t = log(actual_FX_t) - log(fitted_FV_t)

        Args:
            data: DataFrame with log_FX and feature columns.

        Returns:
            Series of gaps (log actual - log fitted) aligned with input index.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        actual_log = data["log_FX"]
        fitted_log = self.fair_value(data)

        # Align indices
        common_idx = actual_log.dropna().index.intersection(
            fitted_log.dropna().index
        )

        gap = actual_log.loc[common_idx] - fitted_log.loc[common_idx]

        return gap

    def summary(self) -> dict:
        """Return model summary statistics."""
        return {
            "alpha": self.alpha_,
            "coefficients": self.coefficients_.to_dict() if self.coefficients_ is not None else None,
            "std_errors": self.std_errors_.to_dict() if self.std_errors_ is not None else None,
            "coint_pvalue": self.coint_pvalue_,
            "adf_pvalue": getattr(self, "_adf_y", None),
            "n_obs": len(self.residuals_) if self.residuals_ is not None else None,
            "resid_std": float(self.residuals_.std()) if self.residuals_ is not None else None,
        }