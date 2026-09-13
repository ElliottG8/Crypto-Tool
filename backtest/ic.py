"""
Information coefficient: daily cross-sectional Spearman rank correlation
between score and (demeaned) forward return, then its mean and a
Newey-West-corrected confidence interval across dates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.newey_west import newey_west_se

DEFAULT_MIN_ASSETS = 5  # a rank correlation over fewer assets than this is too noisy to mean anything
Z_95 = 1.959963984540054  # two-sided 95% normal critical value


def daily_spearman_ic(score_panel: pd.DataFrame, return_panel: pd.DataFrame, min_assets: int = DEFAULT_MIN_ASSETS) -> pd.Series:
    """Spearman correlation between score and return, independently for
    each date, using only assets present (non-NaN) in BOTH panels that
    date. A date with fewer than `min_assets` such assets is NaN — it is
    not scored as zero, and it is excluded from the mean, not counted as
    "no relationship that day."

    Vectorized across all dates at once (rank each row, then row-wise
    Pearson correlation of the ranks — the definition of Spearman's rho)
    rather than one scipy call per date: with a null distribution built
    from hundreds of reshuffles, a per-date Python/scipy loop turns a
    backtest that should take seconds into one that takes many minutes.
    """
    common_dates = score_panel.index.intersection(return_panel.index)
    common_cols = score_panel.columns.union(return_panel.columns)
    s = score_panel.loc[common_dates].reindex(columns=common_cols)
    r = return_panel.loc[common_dates].reindex(columns=common_cols)

    valid = s.notna() & r.notna()
    n_valid = valid.sum(axis=1).to_numpy()

    s_rank = s.where(valid).rank(axis=1).to_numpy()
    r_rank = r.where(valid).rank(axis=1).to_numpy()

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # Dates with zero valid assets (e.g. before any indicator has
        # enough history yet) give an all-NaN row; nanmean over that is a
        # benign NaN, not a bug — the n_valid < min_assets check below
        # already excludes these dates from the result.
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        s_centered = s_rank - np.nanmean(s_rank, axis=1, keepdims=True)
        r_centered = r_rank - np.nanmean(r_rank, axis=1, keepdims=True)
        cov = np.nansum(s_centered * r_centered, axis=1)
        s_var = np.nansum(s_centered**2, axis=1)
        r_var = np.nansum(r_centered**2, axis=1)
        denom = np.sqrt(s_var * r_var)
        corr = np.where(denom > 0, cov / denom, np.nan)

    corr = np.where(n_valid >= min_assets, corr, np.nan)
    return pd.Series(corr, index=common_dates)


@dataclass
class ICResult:
    name: str
    horizon: int
    daily_ic: pd.Series
    mean_ic: float
    newey_west_se: float
    ci_low: float
    ci_high: float
    n_dates: int  # dates with a valid daily IC, actually used in the mean


def summarize_ic(name: str, horizon: int, daily_ic: pd.Series) -> ICResult:
    """Mean IC and its 95% CI via Newey-West with lag = horizon (the
    minimum lag CLAUDE.md calls for, since that's the maximal overlap
    between two forward-return windows of that length)."""
    valid = daily_ic.dropna()
    n = len(valid)
    if n == 0:
        return ICResult(name, horizon, daily_ic, float("nan"), float("nan"), float("nan"), float("nan"), 0)

    mean_ic = float(valid.mean())
    se = newey_west_se(valid.to_numpy(), lag=horizon)
    ci_low = mean_ic - Z_95 * se
    ci_high = mean_ic + Z_95 * se
    return ICResult(name, horizon, daily_ic, mean_ic, se, ci_low, ci_high, n)
