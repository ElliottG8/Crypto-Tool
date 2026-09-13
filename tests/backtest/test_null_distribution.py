import numpy as np
import pandas as pd
import pytest

from backtest.ic import daily_spearman_ic
from backtest.null_distribution import _shuffle_within_dates, compare_to_null, shuffled_null_mean_ics


def test_shuffle_preserves_nan_mask_and_per_row_value_multiset():
    dates = pd.date_range("2024-01-01", periods=8)
    panel = pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, np.nan, 8.0],
            "B": [10.0, np.nan, 30.0, 40.0, 50.0, np.nan, 70.0, 80.0],
            "C": [100.0, 200.0, 300.0, np.nan, np.nan, np.nan, 700.0, 800.0],
        },
        index=dates,
    )
    rng = np.random.default_rng(0)

    shuffled = _shuffle_within_dates(panel, rng)

    assert (panel.isna().to_numpy() == shuffled.isna().to_numpy()).all()
    for i in range(len(dates)):
        original_values = sorted(panel.iloc[i].dropna().to_numpy())
        shuffled_values = sorted(shuffled.iloc[i].dropna().to_numpy())
        assert original_values == shuffled_values


def test_shuffle_actually_permutes_given_enough_rows():
    dates = pd.date_range("2024-01-01", periods=50)
    rng_data = np.random.default_rng(9)
    panel = pd.DataFrame(rng_data.normal(0, 1, (50, 5)), index=dates, columns=list("ABCDE"))

    shuffled = _shuffle_within_dates(panel, np.random.default_rng(1))

    # values are the same per row (just reordered), so it's not literally
    # the identity permutation on essentially every row
    identical_rows = (panel.to_numpy() == shuffled.to_numpy()).all(axis=1).sum()
    assert identical_rows < len(dates)


def test_shuffle_preserves_available_assets_and_dates():
    dates = pd.date_range("2024-01-01", periods=3)
    score = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [2.0, 3.0, 4.0], "C": [3.0, 4.0, 5.0]}, index=dates)
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.02, 0.01, 0.04], "C": [0.03, 0.02, 0.05]}, index=dates)

    draws = shuffled_null_mean_ics(score, returns, n_shuffles=5, min_assets=2, seed=0)
    assert len(draws) == 5
    assert not np.isnan(draws).any()


def test_null_distribution_centers_near_zero_for_unrelated_score_and_return():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=60)
    symbols = ["A", "B", "C", "D", "E", "F"]
    score = pd.DataFrame(rng.normal(0, 1, (60, 6)), index=dates, columns=symbols)
    returns = pd.DataFrame(rng.normal(0, 1, (60, 6)), index=dates, columns=symbols)  # independent of score

    real_ic = daily_spearman_ic(score, returns, min_assets=4).mean()
    draws = shuffled_null_mean_ics(score, returns, n_shuffles=200, min_assets=4, seed=1)
    comparison = compare_to_null(real_ic, draws)

    # unrelated score should land comfortably inside the null distribution
    assert 5 < comparison.percentile_of_real < 95
    assert comparison.two_sided_p_value > 0.05


def test_null_distribution_real_signal_lands_at_extreme_percentile():
    dates = pd.date_range("2024-01-01", periods=40)
    symbols = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.normal(0, 1, (40, 5)), index=dates, columns=symbols)
    score = returns.rank(axis=1)  # perfect signal by construction, IC=1.0 every day

    real_ic = daily_spearman_ic(score, returns, min_assets=4).mean()
    draws = shuffled_null_mean_ics(score, returns, n_shuffles=200, min_assets=4, seed=2)
    comparison = compare_to_null(real_ic, draws)

    assert real_ic == pytest.approx(1.0)
    assert comparison.percentile_of_real > 99.0
    assert comparison.two_sided_p_value < 0.05


def test_compare_to_null_p_value_never_exactly_zero():
    draws = np.random.default_rng(0).normal(0, 0.01, 500)
    comparison = compare_to_null(real_mean_ic=10.0, null_draws=draws)  # absurdly extreme
    assert comparison.two_sided_p_value > 0.0
