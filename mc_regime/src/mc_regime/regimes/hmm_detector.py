"""HMM-based regime detector."""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class HMMSdetector:
    """Hidden Markov Model regime detector.

    Constructs 5 input features from FRED and FX data, fits a Gaussian HMM,
    and uses Viterbi decoding to predict regime states.

    Features:
        The 5 HMM input features (InflationDiff, RealRateDiff, GDPDiff, RealVol, Carry)
        are computed from the FRED columns corresponding to the base/quote currencies
        of the configured pair (default: EURUSD).
    """

    def __init__(
        self,
        fred_levels_path: str,
        fx_path: str,
        pair: str = "EURUSD",
        fred_yoy_path: Optional[str] = None
    ):
        """Initialize the detector.

        Args:
            fred_levels_path: Path to FRED levels CSV
            fx_path: Path to EUR/USD hourly parquet
            pair: Currency pair (EURUSD, GBPUSD, USDJPY)
            fred_yoy_path: Optional path to FRED YoY CSV (computed if not provided)
        """
        self.fred_levels_path = fred_levels_path
        self.fx_path = fx_path
        self.pair = pair.upper()
        self.fred_yoy_path = fred_yoy_path

        self._fred_levels: Optional[pd.DataFrame] = None
        self._fx_hourly: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[GaussianHMM] = None
        self._n_states: int = 2
        self._states: Optional[np.ndarray] = None

    def _load_data(self) -> None:
        """Load FRED and FX data."""
        self._fred_levels = pd.read_csv(self.fred_levels_path)
        self._fred_levels["date"] = pd.to_datetime(self._fred_levels["date"])
        self._fred_levels = self._fred_levels.set_index("date")

        # Also load YoY data (or compute it)
        if self.fred_yoy_path:
            self._fred_yoy = pd.read_csv(self.fred_yoy_path)
            self._fred_yoy["date"] = pd.to_datetime(self._fred_yoy["date"])
            self._fred_yoy = self._fred_yoy.set_index("date")
        else:
            # Compute YoY from levels
            self._fred_yoy = self._compute_fred_yoy()

        # Load FX data
        self._fx_hourly = pd.read_parquet(self.fx_path)
        if "timestamp" in self._fx_hourly.columns:
            self._fx_hourly = self._fx_hourly.set_index("timestamp")
        self._fx_hourly.index = pd.to_datetime(self._fx_hourly.index)
        # Strip timezone to match FRED data (if present)
        if self._fx_hourly.index.tz is not None:
            self._fx_hourly.index = self._fx_hourly.index.tz_convert(None)

    def _compute_fred_yoy(self) -> pd.DataFrame:
        """Compute YoY growth rates from FRED levels.

        Computes on monthly data to handle gaps from weekends/holidays.
        """
        # Resample to monthly first
        monthly = self._fred_levels.resample("ME").last()
        # YoY = pct change over 12 months
        yoy = monthly.pct_change(periods=12) * 100
        return yoy

    def _build_features(self, pair: str = None) -> pd.DataFrame:
        """Build the 5 HMM input features.

        Args:
            pair: Currency pair (EURUSD, GBPUSD, USDJPY). If None, uses self.pair.

        Returns:
            DataFrame with features: InflationDiff, RealRateDiff, GDPDiff, RealVol, Carry.
        """
        # Resolve pair - use self.pair if not provided
        if pair is None:
            pair = getattr(self, "pair", "EURUSD")
        pair_upper = pair.upper()
        if pair_upper == "EURUSD":
            base_ccy, quote_ccy = "EUR", "USD"
        elif pair_upper == "GBPUSD":
            base_ccy, quote_ccy = "GBP", "USD"
        elif pair_upper == "USDJPY":
            base_ccy, quote_ccy = "USD", "JPY"
        else:
            raise ValueError(f"Unsupported pair: {pair}")

        # Map currency to column names
        rate_col_map = {
            "EUR": "EUR_Policy Rate",
            "GBP": "GBP_Policy Rate",
            "JPY": "JPY_Policy Rate",
            "USD": "USD_Fed Funds",
        }
        base_rate_col = rate_col_map[base_ccy]
        quote_rate_col = rate_col_map[quote_ccy]
        base_cpi = f"{base_ccy}_CPI"
        quote_cpi = f"{quote_ccy}_CPI"
        base_gdp = f"{base_ccy}_GDP"
        quote_gdp = f"{quote_ccy}_GDP"

        # Resample FRED data to monthly (month-end)
        fred_m = self._fred_levels.resample("ME").last()
        fred_yoy_m = self._fred_yoy.resample("ME").last()

        # Feature 1: InflationDiff (YoY CPI difference)
        # Determine column names based on what's available
        base_cpi_yoy_col = f"{base_ccy}_CPI_YoY"
        quote_cpi_yoy_col = f"{quote_ccy}_CPI_YoY"
        if base_cpi_yoy_col in fred_yoy_m.columns and quote_cpi_yoy_col in fred_yoy_m.columns:
            inflation_diff = fred_yoy_m[base_cpi_yoy_col] - fred_yoy_m[quote_cpi_yoy_col]
        else:
            inflation_diff = fred_yoy_m[base_cpi] - fred_yoy_m[quote_cpi]

        # Feature 2: RealRateDiff
        # Real rate = nominal - inflation
        if base_cpi_yoy_col in fred_yoy_m.columns and quote_cpi_yoy_col in fred_yoy_m.columns:
            base_cpi_col = base_cpi_yoy_col
            quote_cpi_col = quote_cpi_yoy_col
        else:
            base_cpi_col = base_cpi
            quote_cpi_col = quote_cpi
        base_real = self._fred_levels[base_rate_col].resample("ME").last() - fred_yoy_m[base_cpi_col]
        quote_real = self._fred_levels[quote_rate_col].resample("ME").last() - fred_yoy_m[quote_cpi_col]
        real_rate_diff = base_real - quote_real

        # Feature 3: GDPDiff (GDP YoY difference)
        base_gdp_yoy_col = f"{base_ccy}_GDP_YoY"
        quote_gdp_yoy_col = f"{quote_ccy}_GDP_YoY"
        if base_gdp_yoy_col in fred_yoy_m.columns and quote_gdp_yoy_col in fred_yoy_m.columns:
            gdp_diff = fred_yoy_m[base_gdp_yoy_col] - fred_yoy_m[quote_gdp_yoy_col]
        else:
            gdp_diff = fred_yoy_m[base_gdp] - fred_yoy_m[quote_gdp]

        # Feature 4: RealVol (annualized 3-month rolling std of log returns)
        # Resample FX to daily first
        fx_daily = self._fx_hourly["close"].resample("D").last().dropna()
        log_returns = np.log(fx_daily / fx_daily.shift(1))
        # 3-month rolling std (approx 63 days)
        rolling_std = log_returns.rolling(window=63, min_periods=20).std()
        # Annualize (sqrt of 252 trading days)
        real_vol = rolling_std * np.sqrt(252)
        # Resample to monthly
        real_vol_m = real_vol.resample("ME").mean()

        # Feature 5: Carry (rate differential)
        # Using levels data: base_rate - quote_rate
        carry = self._fred_levels[base_rate_col].resample("ME").last() - self._fred_levels[quote_rate_col].resample("ME").last()

        # Combine into a single DataFrame with proper index alignment
        # Find common dates where all features have valid data
        # Build aligned arrays by iterating through dates

        # Start with dates that have all features (intersection approach)
        valid_dates = inflation_diff.dropna().index.intersection(real_rate_diff.dropna().index)
        valid_dates = valid_dates.intersection(gdp_diff.dropna().index)
        valid_dates = valid_dates.intersection(real_vol_m.dropna().index)
        valid_dates = valid_dates.intersection(carry.dropna().index)

        # Reindex all features to common dates
        features = pd.DataFrame(index=valid_dates)
        features["InflationDiff"] = inflation_diff.loc[valid_dates]
        features["RealRateDiff"] = real_rate_diff.loc[valid_dates]
        features["GDPDiff"] = gdp_diff.loc[valid_dates]
        features["RealVol"] = real_vol_m.loc[valid_dates]
        features["Carry"] = carry.loc[valid_dates]

        # Replace inf with NaN, then drop rows with NaN
        # (v13 fix: GBP/JPY features can produce inf when the underlying
        # FRED series has zero std over a window; downstream StandardScaler
        # would crash on inf.)
        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        # Z-standardize
        self._scaler = StandardScaler()
        features_scaled = self._scaler.fit_transform(features)
        self._features = pd.DataFrame(
            features_scaled,
            index=features.index,
            columns=features.columns
        )

        return self._features

    def _fit_hmm_bic(self, k_values: list = None) -> Tuple[int, GaussianHMM]:
        """Fit HMM with BIC-based state selection.

        Args:
            k_values: List of k values to try (default: [2, 3, 4, 5])

        Returns:
            Tuple of (best_k, best_model)
        """
        if k_values is None:
            k_values = [2, 3, 4, 5]

        X = self._features.values
        n_samples, n_features = X.shape
        bic_scores = {}

        for k in k_values:
            model = GaussianHMM(
                n_components=k,
                covariance_type="diag",
                random_state=42
            )
            model.fit(X)

            # Compute log-likelihood and BIC
            log_likelihood = model.score(X)
            # BIC = -2 * log_likelihood + k * log(n) * n_params
            # For diag covariance: k states + k*n_features means + k*n_features variances
            n_params = k - 1 + k * n_features * 2
            bic = -2 * log_likelihood + n_params * np.log(n_samples)
            bic_scores[k] = bic

            print(f"k={k}: log_likelihood={log_likelihood:.2f}, BIC={bic:.2f}")

        best_k = min(bic_scores, key=bic_scores.get)
        print(f"BIC-selected k={best_k}")

        # Refit with best k
        best_model = GaussianHMM(
            n_components=best_k,
            covariance_type="diag",
            random_state=42
        )
        best_model.fit(X)

        self._n_states = best_k
        self._model = best_model

        return best_k, best_model

    def _predict_states(self, X: np.ndarray = None) -> np.ndarray:
        """Predict hidden states using Viterbi decoding.

        Args:
            X: Feature array (uses self._features if None)

        Returns:
            Array of predicted state indices
        """
        if X is None:
            X = self._features.values

        # Use Viterbi for most likely state sequence
        states = self._model.predict(X)
        self._states = states

        return states

    def fit_predict(self, k_values: list = None) -> Tuple[int, np.ndarray]:
        """Fit the HMM and predict states.

        Args:
            k_values: List of k values to try

        Returns:
            Tuple of (n_states, predicted_states)
        """
        self._load_data()
        self._build_features()
        self._fit_hmm_bic(k_values)
        states = self._predict_states()

        return self._n_states, states

    @property
    def features(self) -> pd.DataFrame:
        """Get the feature DataFrame."""
        return self._features

    @property
    def states(self) -> np.ndarray:
        """Get predicted states."""
        return self._states

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self._n_states

    @property
    def model(self) -> GaussianHMM:
        """Fitted HMM model."""
        return self._model

    def get_states_series(self) -> pd.Series:
        """Get states as a Series indexed by dates."""
        if self._states is None:
            raise ValueError("Must call fit_predict first")
        return pd.Series(
            self._states,
            index=self._features.index
        )


def create_detector(
    fred_path: str = "/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_levels.csv",
    fx_path: str = "/root/research/MC-Regime/mc_regime/outputs/data/EURUSD_1h.parquet",
    fred_yoy_path: str = "/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_yoy.csv"
) -> HMMSdetector:
    """Factory function to create a configured detector."""
    return HMMSdetector(
        fred_levels_path=fred_path,
        fx_path=fx_path,
        fred_yoy_path=fred_yoy_path
    )