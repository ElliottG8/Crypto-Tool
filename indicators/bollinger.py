"""Bollinger Bands and bandwidth."""

from __future__ import annotations

import pandas as pd

from indicators.moving_averages import sma


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands(window, num_std).

    middle = SMA(window)
    std    = population standard deviation (ddof=0) over the same window
             — Bollinger's original definition uses the N divisor, not N-1.
    upper  = middle + num_std * std
    lower  = middle - num_std * std

    NaN until `window` consecutive, complete observations are available
    (inherited from sma's min_periods=window and rolling.std's identical
    NaN-count behavior).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std

    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower}, index=series.index)


def bollinger_bandwidth(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """(upper - lower) / middle — band width normalised by the middle band."""
    bands = bollinger_bands(series, window=window, num_std=num_std)
    return (bands["upper"] - bands["lower"]) / bands["middle"]
