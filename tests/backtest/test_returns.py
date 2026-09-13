import numpy as np
import pandas as pd
import pytest

from backtest.returns import demean_by_universe, forward_returns


def test_forward_return_exact_value():
    dates = pd.date_range("2024-01-01", periods=5)
    close = pd.DataFrame({"A": [100.0, 110.0, 90.0, 120.0, 150.0]}, index=dates)
    result = forward_returns(close, horizon=2)

    # idx0: (90/100)-1 = -0.10 ; idx1: (120/110)-1 = 0.0909...; idx2: (150/90)-1=0.6667
    assert result["A"].iloc[0] == pytest.approx(-0.10)
    assert result["A"].iloc[1] == pytest.approx(120 / 110 - 1)
    assert result["A"].iloc[2] == pytest.approx(150 / 90 - 1)
    # last two rows have no future price 2 steps ahead -> NaN, not fabricated
    assert result["A"].iloc[3:].isna().all()


def test_forward_return_uses_only_price_at_t_and_t_plus_horizon():
    """Changing a price strictly BEFORE t must not change forward_return[t]
    (it only needs close[t] and close[t+horizon]); changing the price
    exactly at t+horizon MUST change it (confirms correct alignment, not
    an off-by-one that reads t+horizon-1 or t+horizon+1)."""
    dates = pd.date_range("2024-01-01", periods=7)
    close = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]}, index=dates)
    horizon = 3
    baseline = forward_returns(close, horizon)

    perturbed_before = close.copy()
    perturbed_before.loc[dates[0], "A"] = 999.0  # before t=2 (the row we check)
    result_before = forward_returns(perturbed_before, horizon)
    assert result_before["A"].iloc[2] == pytest.approx(baseline["A"].iloc[2])

    perturbed_future = close.copy()
    perturbed_future.loc[dates[2 + horizon], "A"] = 999.0  # exactly t+horizon for t=2
    result_future = forward_returns(perturbed_future, horizon)
    assert result_future["A"].iloc[2] != pytest.approx(baseline["A"].iloc[2])
    assert result_future["A"].iloc[2] == pytest.approx(999.0 / close["A"].iloc[2] - 1)

    perturbed_after = close.copy()
    perturbed_after.loc[dates[2 + horizon + 1], "A"] = 999.0  # one step past t+horizon
    result_after = forward_returns(perturbed_after, horizon)
    assert result_after["A"].iloc[2] == pytest.approx(baseline["A"].iloc[2])


def test_forward_return_handles_asset_absent_on_a_date():
    dates = pd.date_range("2024-01-01", periods=5)
    close = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0, 104.0], "B": [np.nan, np.nan, 50.0, 51.0, 52.0]}, index=dates
    )
    result = forward_returns(close, horizon=2)
    # B absent at idx0,1 -> forward return there must be NaN, not computed
    # against some fabricated baseline.
    assert result["B"].iloc[0:2].isna().all()
    assert result["B"].iloc[2] == pytest.approx(52 / 50 - 1)


def test_demean_by_universe_subtracts_equal_weight_mean_ignoring_absent_assets():
    dates = pd.date_range("2024-01-01", periods=2)
    fwd = pd.DataFrame({"A": [0.10, 0.20], "B": [0.20, np.nan], "C": [0.30, 0.40]}, index=dates)
    result = demean_by_universe(fwd)

    # date0: mean(0.10,0.20,0.30)=0.20
    assert result.loc[dates[0], "A"] == pytest.approx(0.10 - 0.20)
    assert result.loc[dates[0], "B"] == pytest.approx(0.20 - 0.20)
    assert result.loc[dates[0], "C"] == pytest.approx(0.30 - 0.20)

    # date1: B absent -> equal-weight mean uses only A,C: mean(0.20,0.40)=0.30
    assert result.loc[dates[1], "A"] == pytest.approx(0.20 - 0.30)
    assert np.isnan(result.loc[dates[1], "B"])
    assert result.loc[dates[1], "C"] == pytest.approx(0.40 - 0.30)


def test_forward_returns_rejects_invalid_horizon():
    dates = pd.date_range("2024-01-01", periods=3)
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
    with pytest.raises(ValueError):
        forward_returns(close, horizon=0)
