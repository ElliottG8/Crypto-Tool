"""A minimal fake CCXT exchange for offline tests. No network calls, ever."""

from __future__ import annotations

from data.timeframes import TIMEFRAME_MS


class FakeExchange:
    """Mimics just enough of ccxt.Exchange for data/ohlcv.py to run against.

    candles_by_symbol: {"BTC/USDT": [[ts, o, h, l, c, v], ...]} sorted by ts.
    Records every fetch_ohlcv call in `.calls` so tests can assert on what
    was actually requested (e.g. that a delta fetch only asked for new data).
    """

    def __init__(self, markets: set[str], candles_by_symbol: dict[str, list[list]], now_ms: int):
        self._markets = markets
        self.candles_by_symbol = candles_by_symbol
        self._now_ms = now_ms
        self.calls: list[dict] = []

    @property
    def markets(self):
        return {m: {} for m in self._markets}

    def load_markets(self):
        return self.markets

    def milliseconds(self) -> int:
        return self._now_ms

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append({"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit})
        if symbol not in self.candles_by_symbol:
            return []
        candles = [c for c in self.candles_by_symbol[symbol] if c[0] >= since]
        candles.sort(key=lambda c: c[0])
        return candles[:limit]


def make_daily_candles(start_ts_ms: int, n: int, price: float = 100.0) -> list[list]:
    """n consecutive daily candles starting at start_ts_ms, strictly increasing price."""
    tf_ms = TIMEFRAME_MS["1d"]
    return [
        [start_ts_ms + i * tf_ms, price + i, price + i + 1, price + i - 1, price + i + 0.5, 1000.0 + i]
        for i in range(n)
    ]
