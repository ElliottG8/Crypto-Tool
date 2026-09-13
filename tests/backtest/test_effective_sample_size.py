import numpy as np
import pandas as pd
import pytest

from backtest.effective_sample_size import average_pairwise_correlation, effective_n_assets


def test_average_pairwise_correlation_hand_verified():
    # A, B perfectly co-move (B=2A); C perfectly anti-moves (C=-A).
    # pairwise: corr(A,B)=+1, corr(A,C)=-1, corr(B,C)=-1 -> average = -1/3.
    dates = pd.date_range("2024-01-01", periods=5)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    panel = pd.DataFrame({"A": a, "B": 2 * a, "C": -a}, index=dates)

    result = average_pairwise_correlation(panel)
    assert result == pytest.approx(-1 / 3, rel=1e-9)


def test_average_pairwise_correlation_matches_independent_numpy_calc():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2024-01-01", periods=50)
    panel = pd.DataFrame(rng.normal(0, 1, (50, 4)), index=dates, columns=["A", "B", "C", "D"])

    result = average_pairwise_correlation(panel)

    corr_matrix = np.corrcoef(panel.to_numpy(), rowvar=False)
    n = corr_matrix.shape[0]
    expected = (corr_matrix.sum() - n) / (n * n - n)  # exclude diagonal (all 1s)
    assert result == pytest.approx(expected, rel=1e-9)


def test_effective_n_assets_hand_verified():
    assert effective_n_assets(n_assets=20, avg_pairwise_corr=0.0) == pytest.approx(20.0)
    assert effective_n_assets(n_assets=20, avg_pairwise_corr=1.0) == pytest.approx(1.0)
    assert effective_n_assets(n_assets=5, avg_pairwise_corr=0.5) == pytest.approx(5 / 3, rel=1e-9)


def test_effective_n_assets_clipped_to_valid_range():
    # Strongly negative correlation would push the raw formula above N;
    # clip keeps it sane rather than reporting "more independent assets
    # than assets."
    result = effective_n_assets(n_assets=4, avg_pairwise_corr=-0.9)
    assert 1.0 <= result <= 4.0


def test_effective_n_assets_single_asset_is_itself():
    assert effective_n_assets(n_assets=1, avg_pairwise_corr=0.5) == pytest.approx(1.0)
