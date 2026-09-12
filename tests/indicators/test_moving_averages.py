import numpy as np
import pandas as pd
import pytest

from indicators.moving_averages import ema, sma


def test_sma_exact_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = sma(s, window=3)
    expected = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True)


def test_sma_shorter_than_window_is_all_nan_not_partial():
    s = pd.Series([1.0, 2.0])  # window=3 but only 2 points
    result = sma(s, window=3)
    assert result.isna().all()
    assert len(result) == 2


def test_sma_nan_in_window_makes_output_nan_not_a_wrong_number():
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    result = sma(s, window=3)
    # window ending at idx2 contains the NaN -> NaN, not mean(1,2) or mean(2,4)
    assert np.isnan(result.iloc[2])
    # window ending at idx3 = [2, NaN, 4] -> still NaN
    assert np.isnan(result.iloc[3])
    # window ending at idx4 = [NaN, 4, 5] -> still NaN (NaN still in window)
    assert np.isnan(result.iloc[4])


def test_sma_recovers_once_nan_is_out_of_window():
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    result = sma(s, window=3)
    # window ending at idx5 = [4,5,6], no NaN present -> real number
    assert result.iloc[5] == pytest.approx(5.0)


def test_ema_exact_values_window_3():
    # Hand-verified independently (seed = SMA of first 3, alpha = 2/(3+1) = 0.5):
    # seed@idx2 = mean(1,2,3) = 2; idx3 = 4*0.5+2*0.5=3; idx4=4; ... linear so ema_i = i.
    s = pd.Series([float(x) for x in range(1, 11)])  # 1..10
    result = ema(s, window=3)
    expected = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True, rtol=1e-12)


def test_ema_shorter_than_window_is_all_nan_not_partial():
    s = pd.Series([1.0, 2.0])  # window=3
    result = ema(s, window=3)
    assert result.isna().all()


def test_ema_nan_resets_seed_and_recovers_after_fresh_window():
    # A NaN early on must not be silently folded into the running average;
    # EMA should stay NaN until a fresh, uninterrupted run of `window`
    # valid values appears after the gap.
    s = pd.Series([1.0, 2.0, np.nan, 10.0, 20.0, 30.0, 40.0])
    result = ema(s, window=3)
    # never got 3 clean values before the NaN -> NaN at idx2
    assert np.isnan(result.iloc[2])
    # idx3, idx4 still filling the fresh seed window (10, 20, then need one more)
    assert np.isnan(result.iloc[3])
    assert np.isnan(result.iloc[4])
    # idx5: seed = mean(10,20,30) = 20
    assert result.iloc[5] == pytest.approx(20.0)
    # idx6: alpha=0.5 -> 40*0.5 + 20*0.5 = 30
    assert result.iloc[6] == pytest.approx(30.0)
