"""Test that HMMSdetector is pair-aware (EURUSD, GBPUSD, USDJPY)."""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from mc_regime.regimes.hmm_detector import HMMSdetector


@pytest.fixture
def fred_panel():
    # Need 36 months minimum for YoY computation (12 months for YoY + overlap with FX)
    dates = pd.date_range("2019-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"date": dates})
    df["EUR_CPI"] = 100 + np.arange(36) * 0.2
    df["EUR_GDP"] = 1000 + np.arange(36) * 1.0
    df["EUR_Policy Rate"] = 0.5
    df["USD_CPI"] = 100 + np.arange(36) * 0.3
    df["USD_GDP"] = 1500 + np.arange(36) * 1.5
    df["USD_Fed Funds"] = 2.0
    df["GBP_CPI"] = 100 + np.arange(36) * 0.25
    df["GBP_GDP"] = 1200 + np.arange(36) * 1.2
    df["GBP_Policy Rate"] = 0.75
    df["JPY_CPI"] = 100 + np.arange(36) * 0.1
    df["JPY_GDP"] = 800 + np.arange(36) * 0.8
    df["JPY_Policy Rate"] = -0.1
    return df.set_index("date")


def test_hmm_detector_accepts_pair_param():
    """The __init__ must accept a pair parameter and default to EURUSD."""
    # Don't actually call fit_predict — just test the signature
    import inspect
    sig = inspect.signature(HMMSdetector.__init__)
    assert "pair" in sig.parameters
    assert sig.parameters["pair"].default == "EURUSD"


def test_hmm_detector_unsupported_pair_raises(fred_panel, tmp_path):
    """fit_predict with an unsupported pair must raise ValueError."""
    # Create a minimal FRED file
    fred_path = tmp_path / "fred.csv"
    fred_panel.reset_index().to_csv(fred_path, index=False)
    # Create a minimal FX file
    fx_dates = pd.date_range("2020-01-01", periods=100, freq="h")
    fx = pd.DataFrame({"timestamp": fx_dates, "close": np.linspace(1.1, 1.2, 100)})
    fx_path = tmp_path / "fx.parquet"
    fx.to_parquet(fx_path)
    # Patch the _fit_hmm_bic and _predict_states to skip the actual HMM
    det = HMMSdetector(fred_levels_path=str(fred_path), fx_path=str(fx_path), pair="AUDUSD")
    det._load_data()
    with pytest.raises(ValueError, match="Unsupported pair"):
        det._build_features()


def test_hmm_detector_eurusd_build_features_works(fred_panel, tmp_path):
    fred_path = tmp_path / "fred.csv"
    fred_panel.reset_index().to_csv(fred_path, index=False)
    # Need at least 3 months of FX data for RealVol rolling window
    fx_dates = pd.date_range("2020-01-01", periods=100*24, freq="h")  # 100 days
    fx = pd.DataFrame({"timestamp": fx_dates, "close": np.linspace(1.1, 1.2, 100*24)})
    fx_path = tmp_path / "fx.parquet"
    fx.to_parquet(fx_path)
    det = HMMSdetector(fred_levels_path=str(fred_path), fx_path=str(fx_path), pair="EURUSD")
    det._load_data()
    features = det._build_features()
    assert "InflationDiff" in features.columns
    assert "RealRateDiff" in features.columns
    assert "GDPDiff" in features.columns
    assert not features[["InflationDiff", "RealRateDiff", "GDPDiff"]].isna().any().any()


def test_hmm_detector_gbpusd_build_features_works(fred_panel, tmp_path):
    fred_path = tmp_path / "fred.csv"
    fred_panel.reset_index().to_csv(fred_path, index=False)
    # Need at least 3 months of FX data for RealVol rolling window
    fx_dates = pd.date_range("2020-01-01", periods=100*24, freq="h")  # 100 days
    fx = pd.DataFrame({"timestamp": fx_dates, "close": np.linspace(1.3, 1.35, 100*24)})
    fx_path = tmp_path / "fx.parquet"
    fx.to_parquet(fx_path)
    det = HMMSdetector(fred_levels_path=str(fred_path), fx_path=str(fx_path), pair="GBPUSD")
    det._load_data()
    features = det._build_features()
    assert not features[["InflationDiff", "RealRateDiff", "GDPDiff"]].isna().any().any()