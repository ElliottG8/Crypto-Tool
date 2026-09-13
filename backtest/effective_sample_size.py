"""
Effective number of independent assets, from the average pairwise
correlation of asset returns.

Crypto assets move together. If a cross-section nominally has 20 assets
but their returns average pairwise correlation rho, a single date's
cross-sectional IC carries roughly as much independent information as
would come from N_eff = N / (1 + (N-1)*rho) genuinely independent assets
— the standard equicorrelation effective-sample-size formula. Treating
"20 assets x T dates" as 20T independent observations, when N_eff is much
smaller than 20, is exactly the overconfidence CLAUDE.md warns against.

This does not attempt to fold the cross-sectional effect and the
Newey-West time-series correction into one combined number — that joint
estimation is genuinely hard and we are not going to fake precision we
don't have. The two are reported side by side; both push the honest
uncertainty wider than either one shown alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_pairwise_correlation(return_panel: pd.DataFrame) -> float:
    """Mean off-diagonal Pearson correlation between assets' return
    columns, pairwise-complete (each pair uses whatever dates both assets
    have — e.g. a late-listed asset still contributes using its overlap
    with the others, rather than being dropped for having fewer rows)."""
    corr = return_panel.corr(method="pearson", min_periods=2)
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    values = corr.to_numpy()[mask]
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def effective_n_assets(n_assets: int, avg_pairwise_corr: float) -> float:
    """N / (1 + (N-1)*rho), clipped to [1, N] (a negative or NaN rho, or
    numerical noise, must not produce a nonsensical effective N below 1
    or above the nominal count)."""
    if n_assets <= 1:
        return float(n_assets)
    if np.isnan(avg_pairwise_corr):
        return float("nan")
    denominator = 1.0 + (n_assets - 1) * avg_pairwise_corr
    if denominator <= 0:
        return float(n_assets)  # negative-correlation edge case: no evidence of redundancy
    eff = n_assets / denominator
    return float(np.clip(eff, 1.0, n_assets))
