"""
Cross-sectional assembly and ranking.

This is where "scores must be cross-sectional, not absolute" actually
happens: a factor's raw values (in whatever units) get turned into a
percentile rank AGAINST THE OTHER ASSETS ON THE SAME DATE. That is a
row-wise operation on a (date x symbol) panel — it cannot see any other
date, so it cannot leak future information no matter how much history is
in the panel. Expanding/rolling causality is enforced by indicators/ and
scoring/factors.py; this module only ever looks sideways (across assets),
never backward or forward in time.
"""

from __future__ import annotations

import pandas as pd


def build_panel(per_symbol_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Assemble per-symbol factor Series into a wide (date x symbol) panel.

    Dates where a symbol has no value (before its listing, after a gap,
    etc.) are NaN for that symbol — never filled, never assumed to be zero.
    """
    if not per_symbol_series:
        raise ValueError("per_symbol_series must not be empty")
    return pd.DataFrame(per_symbol_series).sort_index()


def cross_sectional_percentile(panel: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank (0-1) across columns (assets), independently for
    each row (date). NaN values are ignored: a date with only 5 of 20
    assets listed ranks those 5 against each other, not against 15
    fabricated peers, and the percentile denominator is 5, not 20.
    """
    return panel.rank(axis=1, pct=True, na_option="keep")
