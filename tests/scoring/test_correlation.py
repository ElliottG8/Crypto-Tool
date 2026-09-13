import numpy as np
import pandas as pd
import pytest

from scoring.correlation import factor_correlation_matrix


def test_correlation_matrix_diagonal_is_one_and_symmetric():
    dates = pd.date_range("2024-01-01", periods=5)
    rng = np.random.default_rng(3)
    panels = {
        "f1": pd.DataFrame(rng.normal(0, 1, (5, 3)), index=dates, columns=["A", "B", "C"]),
        "f2": pd.DataFrame(rng.normal(0, 1, (5, 3)), index=dates, columns=["A", "B", "C"]),
    }
    corr = factor_correlation_matrix(panels)

    assert corr.loc["f1", "f1"] == pytest.approx(1.0)
    assert corr.loc["f2", "f2"] == pytest.approx(1.0)
    assert corr.loc["f1", "f2"] == pytest.approx(corr.loc["f2", "f1"])


def test_correlation_matrix_matches_independent_numpy_calculation():
    # f2 = -f1 exactly (same scale in both columns), so the pooled series
    # is perfectly anti-correlated regardless of how the two columns mix.
    dates = pd.date_range("2024-01-01", periods=4)
    f1 = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]}, index=dates)
    f2 = pd.DataFrame({"A": [-1.0, -2.0, -3.0, -4.0], "B": [-5.0, -6.0, -7.0, -8.0]}, index=dates)

    corr = factor_correlation_matrix({"f1": f1, "f2": f2})

    # Independent check via numpy.corrcoef on the pooled, flattened values.
    pooled_f1 = np.concatenate([f1["A"].to_numpy(), f1["B"].to_numpy()])
    pooled_f2 = np.concatenate([f2["A"].to_numpy(), f2["B"].to_numpy()])
    expected = np.corrcoef(pooled_f1, pooled_f2)[0, 1]

    assert corr.loc["f1", "f2"] == pytest.approx(expected)
    assert corr.loc["f1", "f2"] == pytest.approx(-1.0)


def test_correlation_matrix_pairwise_deletes_missing_cells():
    dates = pd.date_range("2024-01-01", periods=3)
    f1 = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
    f2 = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=dates)  # missing the middle date

    corr = factor_correlation_matrix({"f1": f1, "f2": f2})
    # only dates 0 and 2 are usable for the f1/f2 pair; both perfectly aligned -> corr 1.0
    assert corr.loc["f1", "f2"] == pytest.approx(1.0)


def test_correlation_matrix_rejects_empty_input():
    with pytest.raises(ValueError):
        factor_correlation_matrix({})
