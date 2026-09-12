"""Rolling volume ratio: current volume relative to its own trailing average."""

from __future__ import annotations

import pandas as pd

from indicators.moving_averages import sma


def rolling_volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    """volume[i] / SMA(volume, window)[i].

    The average includes the current bar (a rolling window ending at i,
    not shifted) — that's still only data available at or before i, so it
    is not lookahead. NaN wherever the trailing SMA itself is NaN (fewer
    than `window` complete observations, or a NaN inside the window), or
    where volume[i] itself is NaN.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return volume / sma(volume, window)
