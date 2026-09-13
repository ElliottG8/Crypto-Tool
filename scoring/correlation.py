"""
Empirical correlation matrix between factors.

Per CLAUDE.md's non-negotiable rules: RSI, MACD, moving averages, and
"momentum" are all deterministic functions of the same close-price series,
so several of them agreeing is one observation measured several ways, not
independent confirmation. This module makes that measurable rather than
asserted: every scoring run reports the actual empirical correlation
between every pair of factors, computed on the same cross-sectional
percentile scores the composite is built from (pooling every (date,
symbol) observation together, pairwise-deleting where either factor is
NaN for that cell). A composite without this matrix is not something this
code can produce — scorer.py always returns both together.
"""

from __future__ import annotations

import pandas as pd


def factor_correlation_matrix(factor_score_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pearson correlation between factors' cross-sectional percentile
    scores, pooled across every (date, symbol) cell.

    Because each factor panel here is already a within-date percentile
    rank (see cross_section.cross_sectional_percentile), a Pearson
    correlation between two such columns is the pooled Spearman rank
    correlation between the two factors' raw values.
    """
    if not factor_score_panels:
        raise ValueError("factor_score_panels must not be empty")

    stacked = {name: panel.stack() for name, panel in factor_score_panels.items()}
    combined = pd.concat(stacked, axis=1)  # index: (date, symbol) union; columns: factor names
    return combined.corr()
