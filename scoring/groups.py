"""
Factor grouping and equal-weight aggregation.

Groups follow CLAUDE.md: price/trend structure, volume and participation,
volatility state. Derivatives positioning and on-chain/fundamental are
listed in CLAUDE.md as future groups — deliberately absent here, not
stubbed, since there is no derivatives/on-chain data layer yet.

WEIGHTING — deliberately unfitted. Every factor within a group gets equal
weight, and every group gets equal weight in the composite. No weight here
was chosen, tuned, or backtested; they are 1/N by construction because
CLAUDE.md requires that any fitted weight be disclosed as fitted with a
held-out period, and none has been fitted. If a factor is missing for a
given asset/date (e.g. not enough history yet), the group average is taken
over whichever of its factors ARE available that date — never treated as
zero, never fabricated.
"""

from __future__ import annotations

import pandas as pd

FACTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "price_trend": ("rsi_14", "macd_histogram_pct", "sma50_pct_distance", "sma200_pct_distance"),
    "volume_participation": ("rolling_volume_ratio", "obv_participation"),
    "volatility_state": ("atr_pct", "bollinger_bandwidth", "realised_volatility"),
}

WEIGHTING_NOTE = (
    "Equal-weighted within each group and across groups (1/N, unfitted). "
    "No factor weight has been tuned or selected against any backtest — "
    "per CLAUDE.md, a fitted weight must be disclosed together with the "
    "held-out period it was fitted on, and none is fitted here."
)


def _average_available(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row-wise, cell-wise mean across panels sharing a (date, symbol)
    shape, skipping NaN (a missing factor for one asset/date does not
    pull that cell toward zero, it's just excluded from that cell's mean).
    """
    stacked = pd.concat(panels, axis=1)  # columns: MultiIndex (factor_name, symbol)
    # groupby(axis=1) was removed in pandas 3.0; transpose so the symbol
    # level is on the index, group there, then transpose back.
    return stacked.T.groupby(level=1).mean().T


def compute_group_scores(factor_score_panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """factor_score_panels: {factor_name: cross-sectional percentile panel}.
    Returns {group_name: group score panel}, equal-weight mean within group.
    """
    group_scores: dict[str, pd.DataFrame] = {}
    for group_name, factor_names in FACTOR_GROUPS.items():
        available = {name: factor_score_panels[name] for name in factor_names if name in factor_score_panels}
        if not available:
            continue
        group_scores[group_name] = _average_available(available)
    return group_scores


def compute_composite(group_score_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean across group score panels (not re-ranked — the
    composite is a plain average of already cross-sectional [0,1] group
    scores, so it remains comparable in the same [0,1] sense)."""
    if not group_score_panels:
        raise ValueError("group_score_panels must not be empty")
    return _average_available(group_score_panels)
