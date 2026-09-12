"""Wilder's Relative Strength Index. Pure function: series in, series out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators._smoothing import reseeding_smoothed_average


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI(window), Wilder's original smoothing.

    avg_gain/avg_loss are each seeded by the simple mean of the first
    `window` price changes, then recursed with Wilder's alpha = 1/window.
    A NaN price resets both averages' seeding — see indicators/_smoothing.py.

    Edge cases at a fully-seeded index:
      - avg_loss == 0 and avg_gain == 0 (no movement at all): RSI = 50.
      - avg_loss == 0, avg_gain > 0 (all gains): RSI = 100.
      - avg_gain == 0, avg_loss > 0 (all losses): RSI = 0.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=series.index)

    delta = np.diff(values)  # delta[k] = values[k+1] - values[k]
    gain = np.where(np.isnan(delta), np.nan, np.where(delta > 0, delta, 0.0))
    loss = np.where(np.isnan(delta), np.nan, np.where(delta < 0, -delta, 0.0))

    alpha = 1.0 / window
    avg_gain = reseeding_smoothed_average(gain, window, alpha)
    avg_loss = reseeding_smoothed_average(loss, window, alpha)

    for k in range(len(delta)):
        ag, al = avg_gain[k], avg_loss[k]
        if np.isnan(ag) or np.isnan(al):
            continue
        if ag == 0.0 and al == 0.0:
            out[k + 1] = 50.0
        elif al == 0.0:
            out[k + 1] = 100.0
        elif ag == 0.0:
            out[k + 1] = 0.0
        else:
            rs = ag / al
            out[k + 1] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(out, index=series.index)
