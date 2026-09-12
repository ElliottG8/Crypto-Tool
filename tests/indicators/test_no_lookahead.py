"""
For every indicator: truncating the series after some index i must not
change any already-computed value at or before i. If it does, the
function looked at data from the future to compute a "past" value.

Because every implementation here is a strict left-to-right pass (a
pandas rolling window ending at i, or our own forward loop in
indicators/_smoothing.py), this should hold trivially — this test is the
safety net that catches a regression, not a proof each is written that way.
"""

import numpy as np
import pandas as pd
import pytest

from indicators.atr import atr
from indicators.bollinger import bollinger_bands, bollinger_bandwidth
from indicators.macd import macd
from indicators.moving_averages import ema, sma
from indicators.obv import obv
from indicators.rsi import rsi
from indicators.volatility import realised_volatility
from indicators.volume import rolling_volume_ratio


def _make_synthetic_ohlcv(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    volume = pd.Series(rng.uniform(100, 1000, n))
    return close, high, low, volume


INDICATOR_FNS = {
    "sma": lambda c, h, l, v: sma(c, 5),
    "ema": lambda c, h, l, v: ema(c, 5),
    "rsi": lambda c, h, l, v: rsi(c, 5),
    "macd_line": lambda c, h, l, v: macd(c, fast=3, slow=6, signal=2)["macd"],
    "macd_signal": lambda c, h, l, v: macd(c, fast=3, slow=6, signal=2)["signal"],
    "macd_histogram": lambda c, h, l, v: macd(c, fast=3, slow=6, signal=2)["histogram"],
    "bollinger_middle": lambda c, h, l, v: bollinger_bands(c, window=5)["middle"],
    "bollinger_upper": lambda c, h, l, v: bollinger_bands(c, window=5)["upper"],
    "bollinger_lower": lambda c, h, l, v: bollinger_bands(c, window=5)["lower"],
    "bollinger_bandwidth": lambda c, h, l, v: bollinger_bandwidth(c, window=5),
    "atr": lambda c, h, l, v: atr(h, l, c, window=5),
    "obv": lambda c, h, l, v: obv(c, v),
    "realised_volatility": lambda c, h, l, v: realised_volatility(c, window=5),
    "rolling_volume_ratio": lambda c, h, l, v: rolling_volume_ratio(v, window=5),
}


@pytest.mark.parametrize("name", INDICATOR_FNS.keys())
@pytest.mark.parametrize("cut", [8, 15, 25])
def test_no_lookahead(name, cut):
    close, high, low, volume = _make_synthetic_ohlcv(n=40)
    fn = INDICATOR_FNS[name]

    full_result = fn(close, high, low, volume)
    truncated_result = fn(close.iloc[:cut], high.iloc[:cut], low.iloc[:cut], volume.iloc[:cut])

    pd.testing.assert_series_equal(
        truncated_result.reset_index(drop=True),
        full_result.iloc[:cut].reset_index(drop=True),
        check_names=False,
        check_exact=False,
        rtol=1e-9,
    )
