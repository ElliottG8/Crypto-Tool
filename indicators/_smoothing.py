"""
Shared recursive smoother used by EMA and Wilder-style indicators (RSI, ATR).

True EMA and Wilder's smoothing are the same recursion —
    current = current + alpha * (value - current)
— differing only in alpha (2/(window+1) for EMA, 1/window for Wilder).
Both need `window` consecutive valid values to seed the first output (via
their simple mean) before recursion can start, and both must NOT quietly
keep recursing through a gap in the input: a NaN resets the state so the
smoother reseeds from scratch on the next run of `window` valid values,
rather than folding a missing value into the running average as if it
were zero or otherwise silently producing a number.

Pure, causal, no I/O: a loop over the array in order, using only values at
or before the current index.
"""

from __future__ import annotations

import numpy as np


def reseeding_smoothed_average(values: np.ndarray, window: int, alpha: float) -> np.ndarray:
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n = len(values)
    out = np.full(n, np.nan)
    current: float | None = None
    seed_run: list[float] = []

    for i in range(n):
        v = values[i]
        if np.isnan(v):
            current = None
            seed_run = []
            continue
        if current is None:
            seed_run.append(v)
            if len(seed_run) == window:
                current = sum(seed_run) / window
                out[i] = current
                seed_run = []
            continue
        current = current + alpha * (v - current)
        out[i] = current

    return out
