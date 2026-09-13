import numpy as np
import pytest

from backtest.newey_west import newey_west_se


def test_newey_west_se_exact_hand_computed_values():
    # x = [1,2,3,4]. Hand-verified independently with Fraction arithmetic
    # (mean=2.5, gamma0=5/4, gamma1=5/16):
    #   lag=0: var_of_mean = gamma0/4 = 5/16               -> se = sqrt(5/16)
    #   lag=1: var_of_mean = (gamma0 + 2*0.5*gamma1)/4 = 25/64 -> se = 0.625 exactly
    #   lag=2: var_of_mean = 17/48                          -> se = sqrt(17/48)
    x = np.array([1.0, 2.0, 3.0, 4.0])

    assert newey_west_se(x, lag=0) == pytest.approx(np.sqrt(5 / 16), rel=1e-9)
    assert newey_west_se(x, lag=1) == pytest.approx(0.625, rel=1e-9)
    assert newey_west_se(x, lag=2) == pytest.approx(np.sqrt(17 / 48), rel=1e-9)


def test_newey_west_se_lag_0_equals_naive_iid_standard_error():
    x = np.array([5.0, 2.0, 9.0, 1.0, 7.0])
    naive_se = x.std(ddof=0) / np.sqrt(len(x))
    assert newey_west_se(x, lag=0) == pytest.approx(naive_se, rel=1e-9)


def test_newey_west_se_ge_naive_for_positively_autocorrelated_series():
    # A smooth, trending series has strong positive autocorrelation: the
    # HAC-corrected SE should come out wider than the naive iid estimate.
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 200)) * 0.01  # random walk: highly autocorrelated
    naive_se = x.std(ddof=0) / np.sqrt(len(x))
    hac_se = newey_west_se(x, lag=30)
    assert hac_se > naive_se


def test_newey_west_se_handles_short_series():
    assert np.isnan(newey_west_se(np.array([]), lag=5))
    assert np.isnan(newey_west_se(np.array([1.0]), lag=5))
    # lag longer than the series shouldn't crash — it's clipped internally
    result = newey_west_se(np.array([1.0, 2.0, 3.0]), lag=30)
    assert result >= 0
