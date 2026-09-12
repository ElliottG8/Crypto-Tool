"""On-Balance Volume. Pure function: series in, series out."""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume.

    OBV[0] = volume[0] (the conventional starting baseline — StockCharts'
    definition of the running total for the first period).
    For i > 0:
      close[i] > close[i-1]  -> OBV[i] = OBV[i-1] + volume[i]
      close[i] < close[i-1]  -> OBV[i] = OBV[i-1] - volume[i]
      close[i] == close[i-1] -> OBV[i] = OBV[i-1]

    OBV is a genuine running total, not a fixed-window indicator like the
    others in this package: it has no window to reseed from. So a NaN in
    either input at index i means the true cumulative total from i onward
    is permanently unknown — OBV[i] and every OBV[j] for j > i are NaN.
    Treating a missing bar as "no volume" or "no price change" and
    continuing the sum would be exactly the silently-wrong-number failure
    mode this project forbids; there is no honest way to recover a running
    total across a gap in it.
    """
    if len(close) != len(volume):
        raise ValueError("close and volume must be the same length")

    close_vals = close.to_numpy(dtype=float)
    vol_vals = volume.to_numpy(dtype=float)
    n = len(close_vals)
    out = np.full(n, np.nan)

    if n == 0:
        return pd.Series(out, index=close.index)

    if np.isnan(close_vals[0]) or np.isnan(vol_vals[0]):
        return pd.Series(out, index=close.index)  # poisoned from the start

    out[0] = vol_vals[0]
    poisoned = False
    for i in range(1, n):
        if poisoned or np.isnan(close_vals[i]) or np.isnan(vol_vals[i]) or np.isnan(close_vals[i - 1]):
            poisoned = True
            continue
        if close_vals[i] > close_vals[i - 1]:
            out[i] = out[i - 1] + vol_vals[i]
        elif close_vals[i] < close_vals[i - 1]:
            out[i] = out[i - 1] - vol_vals[i]
        else:
            out[i] = out[i - 1]

    return pd.Series(out, index=close.index)
