"""MACD: fast EMA minus slow EMA, plus its own EMA as a signal line."""

from __future__ import annotations

import pandas as pd

from indicators.moving_averages import ema


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD(fast, slow, signal).

    macd_line = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line

    Built entirely from indicators.moving_averages.ema, so it inherits that
    function's NaN-reseeding and lookback behavior for free: macd_line is
    NaN until the slower EMA seeds (whichever of fast/slow is larger), and
    signal_line is NaN until `signal` consecutive valid macd_line values
    have accumulated after that.
    """
    if fast < 1 or slow < 1 or signal < 1:
        raise ValueError("fast, slow, and signal must all be >= 1")

    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=series.index,
    )
