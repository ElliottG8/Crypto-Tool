"""
Per-symbol factor computation, built on top of indicators/.

CONVERSION AUDIT — every indicator used here, and whether it needed
converting into a scale-invariant (cross-asset-comparable) form:

  rsi_14                  NO CONVERSION.  Already bounded 0-100 regardless
                           of an asset's price level.
  macd_histogram_pct      CONVERTED.  Raw MACD histogram is an absolute
                           price-unit quantity (EMA(fast)-EMA(slow) of
                           price, further smoothed) — a $500 histogram
                           value means nothing without knowing whether the
                           asset trades at $5 or $50,000. Divided by close.
  sma50_pct_distance,
  sma200_pct_distance     CONVERTED.  A raw SMA is a price level, not
                           comparable across assets. Expressed as
                           (close - SMA) / SMA, i.e. % distance of price
                           from the moving average.
  atr_pct                 CONVERTED.  Raw ATR is an absolute price range
                           per bar — not comparable between BTC (ATR in
                           thousands of dollars) and a sub-$1 altcoin.
                           Divided by close.
  bollinger_bandwidth     NO CONVERSION.  Already (upper-lower)/middle —
                           a ratio, comparable as-is.
  realised_volatility     NO CONVERSION.  Computed from log returns, which
                           are already scale-invariant (ln(k*a/k*b) =
                           ln(a/b) for any positive constant k).
  rolling_volume_ratio    NO CONVERSION.  Already volume / its own
                           trailing average — a ratio, comparable as-is.
  obv_participation       CONVERTED.  Raw OBV is a cumulative sum of raw
                           volume — its level is meaningless across assets
                           and even its own history (it's a running total,
                           not mean-reverting). Converted to a rate: the
                           average daily OBV change over a window,
                           expressed as a fraction of that window's average
                           daily volume — "how many typical days' worth of
                           net buying/selling pressure per day, on average."

No I/O here: every function takes pandas Series/DataFrames already in
memory and returns a pandas Series. Nothing reads from data_cache.
"""

from __future__ import annotations

import pandas as pd

from indicators.atr import atr
from indicators.bollinger import bollinger_bandwidth
from indicators.macd import macd
from indicators.moving_averages import sma
from indicators.obv import obv
from indicators.rsi import rsi
from indicators.volatility import realised_volatility
from indicators.volume import rolling_volume_ratio

RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_FAST_WINDOW = 50
MA_SLOW_WINDOW = 200
ATR_WINDOW = 14
BOLLINGER_WINDOW = 20
VOLUME_RATIO_WINDOW = 20
REALISED_VOL_WINDOW = 20
OBV_PARTICIPATION_WINDOW = 20

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing required columns: {missing}")


def macd_histogram_pct(close: pd.Series) -> pd.Series:
    hist = macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)["histogram"]
    return hist / close


def ma_pct_distance(close: pd.Series, window: int) -> pd.Series:
    ma = sma(close, window)
    return (close - ma) / ma


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = ATR_WINDOW) -> pd.Series:
    return atr(high, low, close, window) / close


def obv_participation(close: pd.Series, volume: pd.Series, window: int = OBV_PARTICIPATION_WINDOW) -> pd.Series:
    """Average daily OBV change over `window`, scaled by that window's
    average daily volume. Dimensionless and comparable across assets with
    very different absolute volume levels."""
    obv_series = obv(close, volume)
    avg_daily_change = (obv_series - obv_series.shift(window)) / window
    avg_volume = volume.rolling(window=window, min_periods=window).mean()
    return avg_daily_change / avg_volume


def compute_all_factors(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute every scale-invariant factor for one symbol's OHLCV frame.

    df must have columns open, high, low, close, volume, indexed by date/
    timestamp (ascending). Returns {factor_name: pd.Series}, each aligned
    to df's index.
    """
    _require_columns(df)
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    return {
        "rsi_14": rsi(close, window=RSI_WINDOW),
        "macd_histogram_pct": macd_histogram_pct(close),
        "sma50_pct_distance": ma_pct_distance(close, MA_FAST_WINDOW),
        "sma200_pct_distance": ma_pct_distance(close, MA_SLOW_WINDOW),
        "atr_pct": atr_pct(high, low, close, ATR_WINDOW),
        "bollinger_bandwidth": bollinger_bandwidth(close, window=BOLLINGER_WINDOW),
        "realised_volatility": realised_volatility(close, window=REALISED_VOL_WINDOW),
        "rolling_volume_ratio": rolling_volume_ratio(volume, window=VOLUME_RATIO_WINDOW),
        "obv_participation": obv_participation(close, volume, window=OBV_PARTICIPATION_WINDOW),
    }
