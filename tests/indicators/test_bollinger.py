import math

import numpy as np
import pandas as pd
import pytest

from indicators.bollinger import bollinger_bands, bollinger_bandwidth


def test_bollinger_bands_exact_values():
    # Hand-verified: window=4, num_std=2, population std (ddof=0).
    # prices[0:4] = [10,12,11,13] -> mean=11.5, var=5/4=1.25, std=sqrt(1.25)
    # prices[1:5] = [12,11,13,14] -> mean=12.5, var=5/4=1.25, std=sqrt(1.25)
    prices = pd.Series([10.0, 12.0, 11.0, 13.0, 14.0])
    result = bollinger_bands(prices, window=4, num_std=2.0)

    std = math.sqrt(1.25)
    assert result["middle"].iloc[0:3].isna().all()
    assert result["middle"].iloc[3] == pytest.approx(11.5)
    assert result["upper"].iloc[3] == pytest.approx(11.5 + 2 * std)
    assert result["lower"].iloc[3] == pytest.approx(11.5 - 2 * std)

    assert result["middle"].iloc[4] == pytest.approx(12.5)
    assert result["upper"].iloc[4] == pytest.approx(12.5 + 2 * std)
    assert result["lower"].iloc[4] == pytest.approx(12.5 - 2 * std)


def test_bollinger_bandwidth_exact_value():
    prices = pd.Series([10.0, 12.0, 11.0, 13.0, 14.0])
    result = bollinger_bandwidth(prices, window=4, num_std=2.0)
    std = math.sqrt(1.25)
    expected = (2 * std * 2) / 11.5  # (upper-lower)/middle at idx3
    assert result.iloc[3] == pytest.approx(expected)


def test_bollinger_shorter_than_window_is_all_nan_not_partial():
    prices = pd.Series([1.0, 2.0, 3.0])  # window=4
    result = bollinger_bands(prices, window=4)
    assert result["middle"].isna().all()
    assert result["upper"].isna().all()
    assert result["lower"].isna().all()


def test_bollinger_nan_in_window_makes_output_nan():
    prices = pd.Series([10.0, 12.0, np.nan, 13.0, 14.0])
    result = bollinger_bands(prices, window=4)
    assert np.isnan(result["middle"].iloc[3])
    assert np.isnan(result["upper"].iloc[3])
    assert np.isnan(result["lower"].iloc[3])
