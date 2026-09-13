import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from backtest.ic import daily_spearman_ic, summarize_ic


def test_daily_ic_perfect_positive_rank_correlation():
    """Score is exactly the rank of the return each day -> IC = +1.0 every
    date, so mean IC = 1.0 with zero variance — a known, exact answer."""
    dates = pd.date_range("2024-01-01", periods=4)
    returns = pd.DataFrame(
        {"A": [0.05, 0.01, 0.03, 0.02], "B": [0.02, 0.03, 0.05, 0.01], "C": [0.01, 0.05, 0.01, 0.05], "D": [0.03, 0.02, 0.02, 0.03]},
        index=dates,
    )
    score = returns.rank(axis=1)  # perfectly monotonic with return, by construction

    ic = daily_spearman_ic(score, returns, min_assets=3)
    assert np.allclose(ic.to_numpy(), 1.0)

    summary = summarize_ic("test", horizon=1, daily_ic=ic)
    assert summary.mean_ic == pytest.approx(1.0)
    assert summary.newey_west_se == pytest.approx(0.0, abs=1e-9)


def test_daily_ic_perfect_negative_rank_correlation():
    dates = pd.date_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.05, 0.01, 0.03], "B": [0.02, 0.03, 0.05], "C": [0.01, 0.05, 0.01]}, index=dates)
    score = -returns.rank(axis=1)  # inversely ranked

    ic = daily_spearman_ic(score, returns, min_assets=3)
    assert np.allclose(ic.to_numpy(), -1.0)


def test_daily_ic_matches_independent_scipy_calculation():
    dates = pd.date_range("2024-01-01", periods=1)
    score = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [4.0], "D": [1.5], "E": [2.0]}, index=dates)
    returns = pd.DataFrame({"A": [0.02], "B": [-0.01], "C": [0.05], "D": [0.03], "E": [-0.02]}, index=dates)

    result = daily_spearman_ic(score, returns, min_assets=3)

    expected, _ = spearmanr(score.iloc[0].to_numpy(), returns.iloc[0].to_numpy())
    assert result.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_daily_ic_excludes_absent_asset_not_zero_fills():
    dates = pd.date_range("2024-01-01", periods=1)
    score = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [np.nan]}, index=dates)  # D has no score
    returns = pd.DataFrame({"A": [0.01], "B": [0.02], "C": [0.03], "D": [0.99]}, index=dates)  # D's return would wreck the correlation if included

    result = daily_spearman_ic(score, returns, min_assets=3)

    expected, _ = spearmanr([1.0, 2.0, 3.0], [0.01, 0.02, 0.03])
    assert result.iloc[0] == pytest.approx(expected, rel=1e-9)
    assert result.iloc[0] == pytest.approx(1.0)  # A,B,C are perfectly co-ranked


def test_daily_ic_below_min_assets_is_nan_not_zero():
    dates = pd.date_range("2024-01-01", periods=1)
    score = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [np.nan], "D": [np.nan]}, index=dates)
    returns = pd.DataFrame({"A": [0.01], "B": [0.02], "C": [0.03], "D": [0.04]}, index=dates)

    result = daily_spearman_ic(score, returns, min_assets=3)  # only A,B available (2 < 3)
    assert np.isnan(result.iloc[0])


def test_summarize_ic_empty_series():
    empty = pd.Series(dtype=float)
    summary = summarize_ic("test", horizon=7, daily_ic=empty)
    assert summary.n_dates == 0
    assert np.isnan(summary.mean_ic)
