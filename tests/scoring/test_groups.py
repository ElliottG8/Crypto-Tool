import numpy as np
import pandas as pd
import pytest

from scoring.groups import compute_composite, compute_group_scores


def test_group_score_is_equal_weight_average_of_its_factors():
    dates = pd.date_range("2024-01-01", periods=2)
    # price_trend group = rsi_14, macd_histogram_pct, sma50_pct_distance, sma200_pct_distance
    factor_scores = {
        "rsi_14": pd.DataFrame({"A": [1.0, 0.5]}, index=dates),
        "macd_histogram_pct": pd.DataFrame({"A": [0.0, 0.5]}, index=dates),
        "sma50_pct_distance": pd.DataFrame({"A": [0.5, 0.5]}, index=dates),
        "sma200_pct_distance": pd.DataFrame({"A": [1.0, 0.5]}, index=dates),
    }
    groups = compute_group_scores(factor_scores)

    expected_row0 = (1.0 + 0.0 + 0.5 + 1.0) / 4
    expected_row1 = (0.5 + 0.5 + 0.5 + 0.5) / 4
    assert groups["price_trend"].loc[dates[0], "A"] == pytest.approx(expected_row0)
    assert groups["price_trend"].loc[dates[1], "A"] == pytest.approx(expected_row1)


def test_group_score_averages_only_available_factors_not_zero_fill():
    dates = pd.date_range("2024-01-01", periods=1)
    factor_scores = {
        "rsi_14": pd.DataFrame({"A": [0.8]}, index=dates),
        "macd_histogram_pct": pd.DataFrame({"A": [np.nan]}, index=dates),  # missing this factor
        "sma50_pct_distance": pd.DataFrame({"A": [0.4]}, index=dates),
        "sma200_pct_distance": pd.DataFrame({"A": [0.6]}, index=dates),
    }
    groups = compute_group_scores(factor_scores)

    # average of the 3 AVAILABLE factors (0.8, 0.4, 0.6), not divided by 4
    # and not treating the missing one as 0.
    expected = (0.8 + 0.4 + 0.6) / 3
    assert groups["price_trend"].loc[dates[0], "A"] == pytest.approx(expected)


def test_composite_is_equal_weight_average_of_groups():
    dates = pd.date_range("2024-01-01", periods=1)
    group_scores = {
        "price_trend": pd.DataFrame({"A": [0.9], "B": [0.1]}, index=dates),
        "volume_participation": pd.DataFrame({"A": [0.3], "B": [0.7]}, index=dates),
        "volatility_state": pd.DataFrame({"A": [0.6], "B": [0.4]}, index=dates),
    }
    composite = compute_composite(group_scores)

    assert composite.loc[dates[0], "A"] == pytest.approx((0.9 + 0.3 + 0.6) / 3)
    assert composite.loc[dates[0], "B"] == pytest.approx((0.1 + 0.7 + 0.4) / 3)


def test_compute_composite_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_composite({})
