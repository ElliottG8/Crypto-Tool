"""Average True Range (Wilder). Pure functions: series in, series out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators._smoothing import reseeding_smoothed_average


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max(high-low, |high-prev_close|, |low-prev_close|).

    The first bar has no previous close, so it's defined as just
    high[0]-low[0] (a genuine definitional edge, not a data gap). For every
    later bar, if the previous close is missing (a real data gap) the two
    gap-dependent components are NaN and, with skipna=False, so is the
    result — we do not quietly fall back to high-low and hide the gap.
    """
    prev_close = close.shift(1)
    range1 = high - low
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()

    tr = pd.concat([range1, range2, range3], axis=1).max(axis=1, skipna=False)
    if len(tr) > 0:
        tr.iloc[0] = range1.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR(window): Wilder-smoothed true range.

    Seeded by the simple mean of the first `window` true-range values, then
    recursed with alpha = 1/window. A NaN true-range value (a genuine data
    gap) resets the seed, exactly as in indicators/_smoothing.py.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    tr = true_range(high, low, close)
    alpha = 1.0 / window
    out = reseeding_smoothed_average(tr.to_numpy(dtype=float), window, alpha)
    return pd.Series(out, index=close.index)
