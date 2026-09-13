"""
Forward returns and the equal-weight-universe demeaning control.

No I/O: takes a wide (date x symbol) close-price panel already in memory.
"""

from __future__ import annotations

import pandas as pd


def forward_returns(close_panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """forward_return[t, symbol] = close[t+horizon, symbol] / close[t, symbol] - 1.

    close_panel must have a regular, contiguous date index (one row per
    calendar day the universe trades, as produced by scoring.cross_section.
    build_panel over daily OHLCV). shift(-horizon) pulls each symbol's
    price `horizon` rows in the future back onto row t — it never reaches
    backward, and every row past the last `horizon` rows of history (and
    every symbol not yet listed, or gapped, on either end of the pair) is
    NaN rather than a fabricated number.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    future_close = close_panel.shift(-horizon)
    return future_close / close_panel - 1.0


def demean_by_universe(forward_return_panel: pd.DataFrame) -> pd.DataFrame:
    """Subtract, for each date, the equal-weight mean forward return across
    whatever assets are actually present that date (pandas' mean(skipna=True)
    already ignores NaN and divides by the count of assets present — it
    never divides by the full nominal universe size). This is the control
    against measuring market beta instead of cross-sectional selection
    skill: without it, a rising market makes every asset's score look
    "predictive" of a positive return.
    """
    equal_weight_return = forward_return_panel.mean(axis=1, skipna=True)
    return forward_return_panel.sub(equal_weight_return, axis=0)
