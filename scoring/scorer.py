"""
Top-level scoring orchestration: raw OHLCV panels in, ScoringResult out.

No I/O — the caller (a script, not this layer) is responsible for loading
OHLCV from data_cache and handing this function plain DataFrames already
in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scoring.correlation import factor_correlation_matrix
from scoring.cross_section import build_panel, cross_sectional_percentile
from scoring.factors import compute_all_factors
from scoring.groups import WEIGHTING_NOTE, compute_composite, compute_group_scores


@dataclass
class ScoringResult:
    factor_scores: dict[str, pd.DataFrame]  # cross-sectional percentile per factor, date x symbol
    group_scores: dict[str, pd.DataFrame]  # equal-weight group score, date x symbol
    composite: pd.DataFrame  # equal-weight composite of groups, date x symbol
    correlation_matrix: pd.DataFrame  # factor x factor, pooled cross-sectional scores
    weighting_note: str


def score_universe(ohlcv: dict[str, pd.DataFrame]) -> ScoringResult:
    """ohlcv: {symbol: DataFrame[open, high, low, close, volume]}, each
    indexed by date/timestamp ascending. Symbols may have different date
    ranges (different listing dates) — handled as missing, not filled.
    """
    if not ohlcv:
        raise ValueError("ohlcv must not be empty")

    raw_factors_by_symbol = {symbol: compute_all_factors(df) for symbol, df in ohlcv.items()}

    factor_names = next(iter(raw_factors_by_symbol.values())).keys()
    factor_score_panels: dict[str, pd.DataFrame] = {}
    for factor_name in factor_names:
        per_symbol = {symbol: factors[factor_name] for symbol, factors in raw_factors_by_symbol.items()}
        raw_panel = build_panel(per_symbol)
        factor_score_panels[factor_name] = cross_sectional_percentile(raw_panel)

    group_scores = compute_group_scores(factor_score_panels)
    composite = compute_composite(group_scores)
    correlation_matrix = factor_correlation_matrix(factor_score_panels)

    return ScoringResult(
        factor_scores=factor_score_panels,
        group_scores=group_scores,
        composite=composite,
        correlation_matrix=correlation_matrix,
        weighting_note=WEIGHTING_NOTE,
    )
