import numpy as np
import pandas as pd
import pytest

from indicators.macd import macd


def test_macd_exact_values_small_windows():
    # Hand-verified independently with Fraction arithmetic: fast=2, slow=3,
    # signal=2 EMAs (seed-then-recurse, see indicators/_smoothing.py) on
    # prices = 1,2,4,3,5,6,4,7,8,6,9.
    prices = pd.Series([1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 4.0, 7.0, 8.0, 6.0, 9.0])
    result = macd(prices, fast=2, slow=3, signal=2)

    assert result["macd"].iloc[0:2].isna().all()
    assert result["signal"].iloc[0:3].isna().all()
    assert result["histogram"].iloc[0:3].isna().all()

    assert result["macd"].iloc[2] == pytest.approx(5 / 6, rel=1e-9)
    assert result["macd"].iloc[3] == pytest.approx(7 / 18, rel=1e-9)
    assert result["signal"].iloc[3] == pytest.approx(11 / 18, rel=1e-9)
    assert result["histogram"].iloc[3] == pytest.approx(-2 / 9, rel=1e-9)

    assert result["macd"].iloc[6] == pytest.approx(49 / 1944, rel=1e-9)
    assert result["signal"].iloc[6] == pytest.approx(191 / 972, rel=1e-9)
    assert result["histogram"].iloc[6] == pytest.approx(-37 / 216, rel=1e-9)


def test_macd_shorter_than_slow_window_is_all_nan():
    prices = pd.Series([1.0, 2.0])  # slow=26 default needs far more
    result = macd(prices)
    assert result["macd"].isna().all()
    assert result["signal"].isna().all()
    assert result["histogram"].isna().all()


def test_macd_nan_in_price_does_not_silently_produce_a_number():
    prices = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = macd(prices, fast=2, slow=3, signal=2)
    # slow EMA's seed window (idx0-2) contains the NaN -> macd stays NaN there
    assert np.isnan(result["macd"].iloc[2])
