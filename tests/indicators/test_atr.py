import numpy as np
import pandas as pd
import pytest

from indicators.atr import atr, true_range


def test_true_range_exact_values():
    # Hand-verified: TR = max(high-low, |high-prev_close|, |low-prev_close|),
    # first bar has no previous close so it's just high-low.
    high = pd.Series([10.0, 11.0, 10.5, 12.0])
    low = pd.Series([9.0, 9.5, 9.8, 10.5])
    close = pd.Series([9.5, 10.5, 10.0, 11.8])

    result = true_range(high, low, close)
    expected = [1.0, 1.5, 0.7, 2.0]
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-9)


def test_true_range_missing_previous_close_is_nan_not_high_minus_low():
    # A genuine data gap at i-1 (not the first bar) must NOT silently fall
    # back to high-low; it should be NaN because the gap gives no way to
    # know whether there was an overnight jump.
    high = pd.Series([10.0, 11.0, 10.5])
    low = pd.Series([9.0, 9.5, 9.8])
    close = pd.Series([9.5, np.nan, 10.0])

    result = true_range(high, low, close)
    assert np.isnan(result.iloc[2])  # prev_close (idx1) missing


def test_atr_exact_values_window_3():
    # Constructed so high-low equals the desired TR at every bar and the
    # gap terms never exceed it (low held constant, close == low), giving
    # TR = [1.0, 1.5, 2.0, 1.0, 0.5] exactly, matching the Wilder-smoothing
    # scratch calc (seed = mean of first 3, alpha = 1/3 after):
    #   seed@idx2 = 1.5; idx3 = 1.5 + 1/3*(1.0-1.5) = 4/3; idx4 = 19/18.
    low = pd.Series([100.0] * 5)
    high = pd.Series([101.0, 101.5, 102.0, 101.0, 100.5])
    close = pd.Series([100.0] * 5)

    result = atr(high, low, close, window=3)
    assert result.iloc[0:2].isna().all()
    assert result.iloc[2] == pytest.approx(1.5, rel=1e-9)
    assert result.iloc[3] == pytest.approx(4 / 3, rel=1e-9)
    assert result.iloc[4] == pytest.approx(19 / 18, rel=1e-9)


def test_atr_shorter_than_window_is_all_nan():
    high = pd.Series([101.0, 102.0])
    low = pd.Series([100.0, 100.0])
    close = pd.Series([100.5, 101.0])
    result = atr(high, low, close, window=3)
    assert result.isna().all()


def test_atr_nan_in_input_does_not_silently_produce_a_number():
    low = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    high = pd.Series([101.0, 101.5, np.nan, 101.0, 100.5])
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    result = atr(high, low, close, window=3)
    assert np.isnan(result.iloc[2])  # TR undefined here
    assert np.isnan(result.iloc[3])  # seed window (0,1,2) still contains the gap
