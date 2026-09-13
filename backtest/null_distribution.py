"""
Null comparison: shuffle scores within each date, preserving the
cross-sectional structure of returns and of the score distribution, and
rebuild the mean-IC statistic many times. If the real mean IC sits
comfortably inside this distribution, the screener has not demonstrated
any edge over a score that carries no genuine information — CLAUDE.md
requires saying so plainly when that happens, not softening it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.ic import DEFAULT_MIN_ASSETS, daily_spearman_ic


def _shuffle_within_dates(score_panel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute each date's score values across the assets present that
    date. Assets absent that date stay absent (NaN) — shuffling never
    invents a score for an asset that had none, and never mixes scores
    across dates.

    Vectorized across all dates via a double argsort (no per-date Python
    loop — with hundreds of null shuffles over a multi-year daily panel,
    a per-date loop dominates runtime): for each row, `stable_order`
    compacts its valid values into natural column order while `random_order`
    gives a uniformly random ordering of the same valid column positions;
    scattering the former into the latter is exactly a random permutation
    of the valid values among the valid positions, leaving NaN positions
    untouched (a row's invalid slots sort after all valid ones in both
    orderings, so they only ever receive back another NaN).
    """
    arr = score_panel.to_numpy()
    n_rows, n_cols = arr.shape
    valid = ~np.isnan(arr)

    col_index = np.broadcast_to(np.arange(n_cols, dtype=float), arr.shape)
    stable_key = np.where(valid, col_index, np.inf)
    stable_order = np.argsort(stable_key, axis=1)
    natural_valid_values = np.take_along_axis(arr, stable_order, axis=1)

    random_key = np.where(valid, rng.random(arr.shape), np.inf)
    random_order = np.argsort(random_key, axis=1)

    shuffled_values = np.empty_like(arr)
    np.put_along_axis(shuffled_values, random_order, natural_valid_values, axis=1)

    return pd.DataFrame(shuffled_values, index=score_panel.index, columns=score_panel.columns)


def shuffled_null_mean_ics(
    score_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    n_shuffles: int,
    min_assets: int = DEFAULT_MIN_ASSETS,
    seed: int = 0,
) -> np.ndarray:
    """n_shuffles draws of mean(daily IC) computed on a within-date-shuffled
    score against the REAL (unshuffled) returns. This is the null
    hypothesis "the score carries no genuine cross-sectional information,"
    holding everything else about the data — including how correlated the
    assets' actual returns are — fixed."""
    rng = np.random.default_rng(seed)
    draws = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shuffled_scores = _shuffle_within_dates(score_panel, rng)
        daily_ic = daily_spearman_ic(shuffled_scores, return_panel, min_assets=min_assets)
        draws[i] = daily_ic.mean()  # pandas .mean() skips the NaN (excluded) dates
    return draws


@dataclass
class NullComparison:
    real_mean_ic: float
    null_draws: np.ndarray
    null_mean: float
    null_std: float
    percentile_of_real: float  # where real_mean_ic sits in the null distribution, 0-100
    two_sided_p_value: float  # fraction of |null draws| >= |real_mean_ic|, permutation-style


def compare_to_null(real_mean_ic: float, null_draws: np.ndarray) -> NullComparison:
    n = len(null_draws)
    percentile = float(np.mean(null_draws <= real_mean_ic) * 100.0)
    # +1/+1 (add-one) smoothing: the standard permutation-test p-value, which
    # never reports exactly 0 no matter how extreme the real value is.
    p_value = float((np.sum(np.abs(null_draws) >= abs(real_mean_ic)) + 1) / (n + 1))
    return NullComparison(
        real_mean_ic=real_mean_ic,
        null_draws=null_draws,
        null_mean=float(np.mean(null_draws)),
        null_std=float(np.std(null_draws, ddof=1)) if n > 1 else float("nan"),
        percentile_of_real=percentile,
        two_sided_p_value=p_value,
    )
