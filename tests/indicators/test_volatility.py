import math

import numpy as np
import pandas as pd
import pytest

from indicators.volatility import realised_volatility


def test_realised_volatility_exact_values():
    # Hand-verified independently via math.log + statistics.pstdev:
    # closes = 100,102,101,105,103,108
    # log returns (idx1..5): 0.019802627, -0.009852296, 0.038839833,
    #                        -0.019231362, 0.047402239
    # rolling population std over window=3 of those returns.
    closes = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0, 108.0])
    result = realised_volatility(closes, window=3)

    assert result.iloc[0:3].isna().all()
    assert result.iloc[3] == pytest.approx(0.020035394279797586, rel=1e-9)
    assert result.iloc[4] == pytest.approx(0.025453997911151396, rel=1e-9)
    assert result.iloc[5] == pytest.approx(0.029600328948886326, rel=1e-9)


def test_realised_volatility_annualization_factor():
    closes = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0, 108.0])
    raw = realised_volatility(closes, window=3)
    annualized = realised_volatility(closes, window=3, annualization_factor=365.0)
    assert annualized.iloc[3] == pytest.approx(raw.iloc[3] * math.sqrt(365.0), rel=1e-9)


def test_realised_volatility_shorter_than_needed_is_all_nan():
    # window=3 needs 4 closes (3 returns); only 3 given -> 2 returns, all NaN
    closes = pd.Series([100.0, 102.0, 101.0])
    result = realised_volatility(closes, window=3)
    assert result.isna().all()


def test_realised_volatility_nan_close_poisons_touching_returns():
    closes = pd.Series([100.0, 102.0, np.nan, 105.0, 103.0, 108.0, 110.0])
    result = realised_volatility(closes, window=3)
    # log_return[2] and log_return[3] both touch the NaN close -> any
    # rolling window of 3 returns containing either must be NaN.
    assert np.isnan(result.iloc[3])
    assert np.isnan(result.iloc[4])
