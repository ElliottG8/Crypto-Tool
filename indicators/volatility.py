"""Realised volatility: rolling standard deviation of log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realised_volatility(close: pd.Series, window: int, annualization_factor: float | None = None) -> pd.Series:
    """Rolling population std (ddof=0) of log returns over `window` periods.

    log_return[i] = ln(close[i] / close[i-1]), so this needs window+1 valid
    consecutive closes to produce its first non-NaN value: one lost to the
    log-return differencing, then `window` of those returns to fill the
    rolling window (min_periods=window — no partial-window average).

    A NaN close poisons the two log returns that touch it, which in turn
    poisons every rolling window that includes either of those returns —
    handled by pandas' own NaN-count-based min_periods, not a manual check.

    If annualization_factor is given (e.g. 365 for daily bars), the result
    is multiplied by sqrt(annualization_factor).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=window, min_periods=window).std(ddof=0)
    if annualization_factor is not None:
        vol = vol * np.sqrt(annualization_factor)
    return vol
