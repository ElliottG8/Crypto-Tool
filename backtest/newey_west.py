"""
Newey-West (1987) HAC standard error of a sample mean, Bartlett kernel.

30D forward returns computed on consecutive daily dates overlap for 29 of
their 30 days, so the daily IC series is heavily autocorrelated — treating
each day's IC as an independent draw badly understates the standard error
of the mean IC. Newey-West corrects for this directly by adding weighted
autocovariance terms up to a chosen lag, rather than assuming they're zero.
We use lag >= the horizon length (the maximal overlap), per CLAUDE.md.
"""

from __future__ import annotations

import numpy as np


def newey_west_se(x: np.ndarray, lag: int) -> float:
    """Standard error of mean(x) under serial correlation up to `lag`.

        Var(mean) = (1/n) * [gamma_0 + 2 * sum_{l=1}^{L} w_l * gamma_l]
        w_l = 1 - l/(L+1)                      (Bartlett kernel)
        gamma_l = (1/n) * sum_{t=l+1}^{n} (x_t - xbar)(x_{t-l} - xbar)

    Clipped at 0 before taking the square root: with a short series and a
    large lag relative to n, the raw HAC variance estimate can go slightly
    negative, which is a known small-sample artifact of the Bartlett
    kernel, not a meaningful "negative uncertainty."
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")

    effective_lag = min(lag, n - 1)
    x_dm = x - x.mean()
    gamma0 = np.dot(x_dm, x_dm) / n

    variance = gamma0
    for l in range(1, effective_lag + 1):
        weight = 1.0 - l / (effective_lag + 1)
        gamma_l = np.dot(x_dm[l:], x_dm[:-l]) / n
        variance += 2.0 * weight * gamma_l

    variance_of_mean = max(variance, 0.0) / n
    return float(np.sqrt(variance_of_mean))
