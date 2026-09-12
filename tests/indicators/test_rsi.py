import numpy as np
import pandas as pd
import pytest

from indicators.rsi import rsi


def test_rsi_exact_values_window_3():
    # Hand-verified independently with Fraction arithmetic (see task notes):
    # prices: 44, 44.5, 44, 43.5, 44.5, 45, 45.5
    # deltas:      0.5, -0.5, -0.5, 1.0, 0.5, 0.5
    # seed (mean of first 3 gains/losses) at idx3, Wilder alpha=1/3 after.
    prices = pd.Series([44.0, 44.5, 44.0, 43.5, 44.5, 45.0, 45.5])
    result = rsi(prices, window=3)

    assert result.iloc[0:3].isna().all()
    assert result.iloc[3] == pytest.approx(100 / 3, rel=1e-9)
    assert result.iloc[4] == pytest.approx(200 / 3, rel=1e-9)
    assert result.iloc[5] == pytest.approx(2500 / 33, rel=1e-9)
    assert result.iloc[6] == pytest.approx(7700 / 93, rel=1e-9)


def test_rsi_all_gains_is_100():
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rsi(prices, window=3)
    assert result.iloc[3] == pytest.approx(100.0)
    assert result.iloc[4] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    prices = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    result = rsi(prices, window=3)
    assert result.iloc[3] == pytest.approx(0.0)
    assert result.iloc[4] == pytest.approx(0.0)


def test_rsi_no_movement_is_50():
    prices = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    result = rsi(prices, window=3)
    assert result.iloc[3] == pytest.approx(50.0)
    assert result.iloc[4] == pytest.approx(50.0)


def test_rsi_shorter_than_window_plus_one_is_all_nan():
    # window=3 needs 4 prices minimum (3 deltas) for a first value
    prices = pd.Series([1.0, 2.0, 3.0])  # only 2 deltas
    result = rsi(prices, window=3)
    assert result.isna().all()


def test_rsi_nan_in_price_does_not_silently_produce_a_number():
    prices = pd.Series([44.0, 44.5, np.nan, 43.5, 44.5, 45.0, 45.5, 46.0])
    result = rsi(prices, window=3)
    # deltas touching the NaN (idx1->idx2 and idx2->idx3) are NaN, so RSI at
    # idx2 and idx3 must be NaN, not a number computed by ignoring the gap.
    assert np.isnan(result.iloc[2])
    assert np.isnan(result.iloc[3])
