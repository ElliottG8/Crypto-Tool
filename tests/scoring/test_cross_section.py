import numpy as np
import pandas as pd
import pytest

from scoring.cross_section import build_panel, cross_sectional_percentile


def test_build_panel_aligns_and_preserves_missing_as_nan():
    a = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3))
    b = pd.Series([10.0, 20.0], index=pd.date_range("2024-01-02", periods=2))  # starts a day later
    panel = build_panel({"A": a, "B": b})

    assert list(panel.columns) == ["A", "B"] or list(panel.columns) == ["B", "A"]
    assert np.isnan(panel.loc["2024-01-01", "B"])
    assert panel.loc["2024-01-02", "B"] == 10.0


def test_percentile_rank_basic_and_ignores_missing():
    dates = pd.date_range("2024-01-01", periods=2)
    panel = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, np.nan], "C": [3.0, 5.0]}, index=dates)

    result = cross_sectional_percentile(panel)

    # date0: A=1 (lowest, rank 1/3), B=2 (mid, 2/3), C=3 (highest, 3/3)
    assert result.loc[dates[0], "A"] == pytest.approx(1 / 3)
    assert result.loc[dates[0], "B"] == pytest.approx(2 / 3)
    assert result.loc[dates[0], "C"] == pytest.approx(3 / 3)

    # date1: B missing -> percentile computed over {A, C} only (denominator 2, not 3)
    assert np.isnan(result.loc[dates[1], "B"])
    assert result.loc[dates[1], "A"] == pytest.approx(1 / 2)
    assert result.loc[dates[1], "C"] == pytest.approx(2 / 2)


def test_percentile_rank_is_stable_under_monotonic_transform():
    """Rank stability: applying the SAME strictly increasing function to
    every value in the panel must not change any cross-sectional rank,
    since rank only depends on relative order within each date."""
    dates = pd.date_range("2024-01-01", periods=4)
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {"A": rng.normal(0, 1, 4), "B": rng.normal(0, 1, 4), "C": rng.normal(0, 1, 4), "D": rng.normal(0, 1, 4)},
        index=dates,
    )

    original = cross_sectional_percentile(panel)

    monotonic_transform = lambda x: np.exp(x) * 3 + 7  # strictly increasing everywhere
    transformed_panel = panel.map(monotonic_transform)
    transformed = cross_sectional_percentile(transformed_panel)

    pd.testing.assert_frame_equal(original, transformed)


def test_percentile_rank_no_lookahead_across_dates():
    """A row's percentile must depend only on that row — appending more
    rows (dates) to the panel must not change any earlier row's result."""
    dates = pd.date_range("2024-01-01", periods=6)
    rng = np.random.default_rng(5)
    full_panel = pd.DataFrame(
        {"A": rng.normal(0, 1, 6), "B": rng.normal(0, 1, 6), "C": rng.normal(0, 1, 6)}, index=dates
    )

    full_result = cross_sectional_percentile(full_panel)
    truncated_result = cross_sectional_percentile(full_panel.iloc[:3])

    pd.testing.assert_frame_equal(truncated_result, full_result.iloc[:3])


def test_build_panel_rejects_empty_input():
    with pytest.raises(ValueError):
        build_panel({})
