"""
Cross-sectional scoring: combine indicators/ into per-factor, per-group,
and composite scores. No I/O, no fitted weights, no backtest — see
CLAUDE.md's layering: data -> indicators -> scoring -> backtest.
"""

from scoring.correlation import factor_correlation_matrix
from scoring.cross_section import build_panel, cross_sectional_percentile
from scoring.factors import compute_all_factors
from scoring.groups import FACTOR_GROUPS, WEIGHTING_NOTE, compute_composite, compute_group_scores
from scoring.scorer import ScoringResult, score_universe

__all__ = [
    "compute_all_factors",
    "build_panel",
    "cross_sectional_percentile",
    "FACTOR_GROUPS",
    "WEIGHTING_NOTE",
    "compute_group_scores",
    "compute_composite",
    "factor_correlation_matrix",
    "ScoringResult",
    "score_universe",
]
