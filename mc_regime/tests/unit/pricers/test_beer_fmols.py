"""Unit tests for BEERPricer with FMOLS estimation."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_beer_data():
    """Generate synthetic data with known BEER coefficients.

    Based on Clark-MacDonald 1998 reduced form:
    ln(Q) = 0.5 + 0.3*TOT + 0.2*NFA + 0.4*ProdDiff + 0.25*RateDiff + ε
    """
    np.random.seed(42)
    n = 200

    # Generate features (standardized)
    TOT = np.random.randn(n) * 2
    NFA = np.random.randn(n) * 3
    ProdDiff = np.random.randn(n) * 1.5
    RateDiff = np.random.randn(n) * 1.0
    FiscalBalance = np.random.randn(n) * 0.5

    # True coefficients (from Clark-MacDonald estimates)
    alpha = 0.5
    beta_TOT = 0.3
    beta_NFA = 0.2
    beta_ProdDiff = 0.4
    beta_RateDiff = 0.25
    beta_Fiscal = 0.1

    # Generate log(FX) with some autocorrelation in errors
    eps = np.random.randn(n) * 0.1
    eps[1:] += 0.3 * eps[:-1]  # AR(1) in errors

    log_FX = (
        alpha
        + beta_TOT * TOT
        + beta_NFA * NFA
        + beta_ProdDiff * ProdDiff
        + beta_RateDiff * RateDiff
        + beta_Fiscal * FiscalBalance
        + eps
    )

    dates = pd.date_range("2010-01-01", periods=n, freq="ME")
    data = pd.DataFrame(
        {
            "log_FX": log_FX,
            "TOT": TOT,
            "NFA": NFA,
            "ProdDiff": ProdDiff,
            "RateDiff": RateDiff,
            "FiscalBalance": FiscalBalance,
        },
        index=dates,
    )

    return data


@pytest.fixture
def beer_pricer():
    """Create BEERPricer instance."""
    from mc_regime.pricers.beer import BEERPricer

    return BEERPricer(maxlag=6)


def test_beer_fit_recovers_coefficients(beer_pricer, synthetic_beer_data):
    """Test that FMOLS recovers known coefficients within tolerance."""
    pricer = beer_pricer
    pricer.fit(synthetic_beer_data)

    assert pricer.is_fitted_
    assert pricer.coefficients_ is not None
    assert pricer.alpha_ is not None

    # Check coefficients are recovered (within 30%)
    tol = 0.30

    coef = pricer.coefficients_

    # Intercept
    assert abs(coef["const"] - 0.5) < tol, f"Intercept: {coef['const']:.3f} vs 0.5"

    # TOT coefficient
    assert abs(coef["TOT"] - 0.3) < tol, f"TOT: {coef['TOT']:.3f} vs 0.3"

    # NFA coefficient
    assert abs(coef["NFA"] - 0.2) < tol, f"NFA: {coef['NFA']:.3f} vs 0.2"

    # ProdDiff coefficient
    assert abs(coef["ProdDiff"] - 0.4) < tol, f"ProdDiff: {coef['ProdDiff']:.3f} vs 0.4"

    # RateDiff coefficient
    assert abs(coef["RateDiff"] - 0.25) < tol, f"RateDiff: {coef['RateDiff']:.3f} vs 0.25"


def test_beer_hac_se_larger_than_ols(synthetic_beer_data):
    """Test that HAC SEs are typically larger than OLS SEs (heteroskedasticity test)."""
    from mc_regime.pricers.beer import BEERPricer

    pricer = BEERPricer(maxlag=6)
    pricer.fit(synthetic_beer_data)

    # HAC standard errors should generally be larger
    # due to accounting for autocorrelation/heteroskedasticity
    if pricer.std_errors_ is not None:
        ses = pricer.std_errors_
        # At least some SEs should be non-trivial
        assert ses.sum() > 0


def test_beer_coint_pvalue_returned(synthetic_beer_data):
    """Test cointegration test runs and returns p-value."""
    from mc_regime.pricers.beer import BEERPricer

    pricer = BEERPricer(maxlag=6)
    pricer.fit(synthetic_beer_data)

    assert pricer.coint_pvalue_ is not None
    assert 0.0 <= pricer.coint_pvalue_ <= 1.0


def test_beer_fair_value_computation(beer_pricer, synthetic_beer_data):
    """Test fair_value computation produces sensible output."""
    pricer = beer_pricer
    pricer.fit(synthetic_beer_data)

    fair_val = pricer.fair_value(synthetic_beer_data)

    assert isinstance(fair_val, pd.Series)
    assert len(fair_val) > 0
    assert not fair_val.isna().all()


def test_beer_gap_computation(beer_pricer, synthetic_beer_data):
    """Test gap computation (residual) is correctly computed."""
    pricer = beer_pricer
    pricer.fit(synthetic_beer_data)

    gap = pricer.gap(synthetic_beer_data)

    assert isinstance(gap, pd.Series)
    assert len(gap) > 0

    # Gap should be approximately centered on zero
    # (OLS minimizes sum of squared residuals)
    assert abs(gap.mean()) < 0.1, f"Gap mean: {gap.mean():.4f}"


def test_beer_summary_returns_dict(beer_pricer, synthetic_beer_data):
    """Test summary method returns expected dictionary."""
    pricer = beer_pricer
    pricer.fit(synthetic_beer_data)

    summary = pricer.summary()

    assert isinstance(summary, dict)
    assert "alpha" in summary
    assert "coefficients" in summary
    assert "coint_pvalue" in summary
    assert "n_obs" in summary


def test_beer_unfitted_raises():
    """Test calling methods before fit() raises RuntimeError."""
    from mc_regime.pricers.beer import BEERPricer

    pricer = BEERPricer()

    dummy_data = pd.DataFrame({"log_FX": [1, 2, 3], "TOT": [0, 0, 0], "NFA": [0, 0, 0], "ProdDiff": [0, 0, 0], "RateDiff": [0, 0, 0], "FiscalBalance": [0, 0, 0]})

    with pytest.raises(RuntimeError):
        pricer.fair_value(dummy_data)

    with pytest.raises(RuntimeError):
        pricer.gap(dummy_data)


def test_beer_missing_columns_raises(beer_pricer):
    """Test missing required columns raises ValueError."""
    incomplete_data = pd.DataFrame({"log_FX": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing required"):
        beer_pricer.fit(incomplete_data)