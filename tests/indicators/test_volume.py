import numpy as np
import pandas as pd
import pytest

from indicators.volume import rolling_volume_ratio


def test_rolling_volume_ratio_exact_values():
    # window=3, SMA includes the current bar.
    # volume: 10, 20, 30, 60, 30
    # sma@idx2 = mean(10,20,30) = 20      -> ratio = 30/20 = 1.5
    # sma@idx3 = mean(20,30,60) = 36.667  -> ratio = 60/36.667 = 1.636364
    # sma@idx4 = mean(30,60,30) = 40      -> ratio = 30/40 = 0.75
    volume = pd.Series([10.0, 20.0, 30.0, 60.0, 30.0])
    result = rolling_volume_ratio(volume, window=3)

    assert result.iloc[0:2].isna().all()
    assert result.iloc[2] == pytest.approx(1.5)
    assert result.iloc[3] == pytest.approx(60 / (110 / 3))
    assert result.iloc[4] == pytest.approx(0.75)


def test_rolling_volume_ratio_shorter_than_window_is_all_nan():
    volume = pd.Series([10.0, 20.0])
    result = rolling_volume_ratio(volume, window=3)
    assert result.isna().all()


def test_rolling_volume_ratio_nan_in_window_makes_output_nan():
    volume = pd.Series([10.0, np.nan, 30.0, 60.0, 30.0])
    result = rolling_volume_ratio(volume, window=3)
    assert np.isnan(result.iloc[2])  # window (10,NaN,30) -> NaN
    assert np.isnan(result.iloc[3])  # window (NaN,30,60) -> NaN
    assert result.iloc[4] == pytest.approx(30 / 40)  # window (30,60,30), clean
