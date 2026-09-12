import numpy as np
import pandas as pd
import pytest

from indicators.obv import obv


def test_obv_exact_values():
    # close:  10,   11,   10,   10,   12
    # move:    -   up    down  flat  up
    # OBV[0]=volume[0]=100
    # OBV[1]=100+50=150 (up)
    # OBV[2]=150-30=120 (down)
    # OBV[3]=120       (flat)
    # OBV[4]=120+70=190 (up)
    close = pd.Series([10.0, 11.0, 10.0, 10.0, 12.0])
    volume = pd.Series([100.0, 50.0, 30.0, 20.0, 70.0])

    result = obv(close, volume)
    expected = [100.0, 150.0, 120.0, 120.0, 190.0]
    np.testing.assert_allclose(result.to_numpy(), expected)


def test_obv_single_bar_is_its_own_volume():
    close = pd.Series([10.0])
    volume = pd.Series([42.0])
    result = obv(close, volume)
    assert result.iloc[0] == pytest.approx(42.0)


def test_obv_nan_permanently_poisons_the_running_total():
    # OBV is a genuine cumulative sum with no window to reseed from: once
    # one increment is unknown, every later total is unknown too.
    close = pd.Series([10.0, 11.0, np.nan, 13.0, 14.0])
    volume = pd.Series([100.0, 50.0, 30.0, 20.0, 70.0])

    result = obv(close, volume)
    assert result.iloc[0] == pytest.approx(100.0)
    assert result.iloc[1] == pytest.approx(150.0)
    assert result.iloc[2:].isna().all()  # gap and everything after: unknown forever


def test_obv_nan_volume_also_poisons_from_that_point_on():
    close = pd.Series([10.0, 11.0, 12.0, 13.0])
    volume = pd.Series([100.0, np.nan, 30.0, 20.0])

    result = obv(close, volume)
    assert result.iloc[0] == pytest.approx(100.0)
    assert result.iloc[1:].isna().all()


def test_obv_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        obv(pd.Series([1.0, 2.0]), pd.Series([1.0]))
