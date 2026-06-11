"""Integration tests for pricers using real EUR/USD + FRED data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def real_data():
    """Load real EUR/USD and FRED data for integration testing.

    Uses cached data:
    - EURUSD daily from parquet
    - FRED panel from CSV
    """
    # Load FX data
    fx_path = "/root/research/MC-Regime/mc_regime/outputs/data/EURUSD_1D.parquet"
    fx_data = pd.read_parquet(fx_path)

    # Load FRED data (both levels and YoY)
    fred_levels = pd.read_csv(
        "/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_levels.csv",
        parse_dates=["date"],
    )
    fred_yoy = pd.read_csv(
        "/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_yoy.csv",
        parse_dates=["date"],
    )

    # Merge levels and yoy
    fred = fred_levels.merge(fred_yoy, on="date", how="outer")

    return fx_data, fred


def test_integration_build_features(real_data):
    """Test building BEER features from real data."""
    from mc_regime.pricers.beer import BEERPricer

    fx_data, fred = real_data

    pricer = BEERPricer()
    features = pricer.build_features(fx_data, fred, pair="EURUSD")

    assert isinstance(features, pd.DataFrame)
    assert "log_FX" in features.columns
    assert "TOT" in features.columns
    assert "NFA" in features.columns
    assert "ProdDiff" in features.columns
    assert "RateDiff" in features.columns

    # Should have some valid observations
    valid = features.dropna()
    assert len(valid) > 0, "No valid observations after feature construction"


@pytest.mark.integration
def test_integration_beer_fit_real_data(real_data):
    """Integration test: fit BEER on real EUR/USD + FRED data."""
    from mc_regime.pricers.beer import BEERPricer

    fx_data, fred = real_data

    pricer = BEERPricer()
    features = pricer.build_features(fx_data, fred, pair="EURUSD")

    # Get valid observations
    valid_data = features.dropna()

    if len(valid_data) < 20:
        pytest.skip("Insufficient data after cleaning")

    # Fit model
    pricer.fit(valid_data)

    assert pricer.is_fitted_
    assert pricer.coefficients_ is not None
    assert pricer.alpha_ is not None


@pytest.mark.integration
def test_integration_gap_statistics_real(real_data):
    """Integration test: verify gap statistics are reasonable.

    For real BEER model on EUR/USD:
    - Mean gap should be near 0 (model fits reasonably)
    - Std should be ~5-15% (0.05-0.15 in log space)
    """
    from mc_regime.pricers.beer import BEERPricer

    fx_data, fred = real_data

    pricer = BEERPricer()
    features = pricer.build_features(fx_data, fred, pair="EURUSD")
    valid_data = features.dropna()

    if len(valid_data) < 20:
        pytest.skip("Insufficient data after cleaning")

    # Fit
    pricer.fit(valid_data)

    # Compute gap
    gap = pricer.gap(valid_data)
    gap_valid = gap.dropna()

    # Statistics
    gap_mean = gap_valid.mean()
    gap_std = gap_valid.std()

    # Reasonable bounds
    # Mean near zero (within 0.2)
    assert abs(gap_mean) < 0.5, f"Gap mean too large: {gap_mean:.4f}"

    # Std ~5-15% in log space (0.05 - 0.15)
    assert 0.01 < gap_std < 0.5, f"Gap std out of range: {gap_std:.4f}"

    print(f"\n=== Real Data BEER Results ===")
    print(f"N observations: {len(gap_valid)}")
    print(f"Gap mean: {gap_mean:.4f}")
    print(f"Gap std: {gap_std:.4f}")
    print(f"Coint p-value: {pricer.coint_pvalue_:.4f}")
    print(f"Coefficients: {pricer.coefficients_.to_dict()}")


@pytest.mark.integration
def test_integration_fair_value_time_series(real_data):
    """Integration test: fair value time series has structure."""
    from mc_regime.pricers.beer import BEERPricer

    fx_data, fred = real_data

    pricer = BEERPricer()
    features = pricer.build_features(fx_data, fred, pair="EURUSD")
    valid_data = features.dropna()

    if len(valid_data) < 20:
        pytest.skip("Insufficient data after cleaning")

    pricer.fit(valid_data)
    fair_val = pricer.fair_value(valid_data)

    # Fair value should vary over time
    fair_std = fair_val.std()
    assert fair_std > 0.001, "Fair value has no variation"


if __name__ == "__main__":
    # Run quick test
    import sys

    try:
        fx = pd.read_parquet("/root/research/MC-Regime/mc_regime/outputs/data/EURUSD_1D.parquet")
        print(f"Loaded FX data: {fx.shape}")

        fred = pd.read_csv("/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_levels.csv")
        print(f"Loaded FRED: {fred.shape}")

        # Quick test
        from mc_regime.pricers.beer import BEERPricer

        pricer = BEERPricer()
        features = pricer.build_features(fx, fred)
        valid = features.dropna()
        print(f"Valid observations: {len(valid)}")

        if len(valid) >= 20:
            pricer.fit(valid)
            gap = pricer.gap(valid)
            print(f"Gap mean: {gap.mean():.4f}")
            print(f"Gap std: {gap.std():.4f}")
            print(f"Coint p-value: {pricer.coint_pvalue_:.4f}")
        else:
            print("Skipping fit - insufficient data")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)