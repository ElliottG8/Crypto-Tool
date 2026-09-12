"""Simple and exponential moving averages. Pure functions: series in, series out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators._smoothing import reseeding_smoothed_average

# Standard lookbacks used elsewhere in this project. Not enforced here —
# sma/ema accept any window — just the canonical set per CLAUDE.md.
MA_WINDOWS: tuple[int, ...] = (20, 50, 100, 200)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average.

    NaN for every index until `window` consecutive, complete observations
    are available ending at that index. A single NaN inside the window
    makes that window's output NaN too (min_periods=window rather than
    silently averaging over the fewer non-null values pandas would
    otherwise use).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average, seeded by the SMA of the first `window`
    valid values, then recursed with alpha = 2 / (window + 1).

    NaN until seeded. A NaN in the input resets the seed: the next EMA
    value only appears after a fresh, uninterrupted run of `window` valid
    observations.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    alpha = 2.0 / (window + 1)
    values = series.to_numpy(dtype=float)
    out = reseeding_smoothed_average(values, window, alpha)
    return pd.Series(out, index=series.index)
