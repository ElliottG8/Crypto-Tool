import pandas as pd

from data import cache, ohlcv, quality
from data.timeframes import TIMEFRAME_MS
from tests.fakes import FakeExchange

DAY = TIMEFRAME_MS["1d"]
START = 1_600_000_000_000


def _df(timestamps):
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * len(timestamps),
            "high": [1.0] * len(timestamps),
            "low": [1.0] * len(timestamps),
            "close": [1.0] * len(timestamps),
            "volume": [1.0] * len(timestamps),
        }
    )


def test_no_gap_in_contiguous_series():
    df = _df([START + i * DAY for i in range(10)])
    assert quality.detect_gaps(df, "1d") == []


def test_gap_is_detected_and_sized_correctly():
    # Days 0,1,2 present, then a 4-day hole, then day 7.
    timestamps = [START + i * DAY for i in (0, 1, 2, 7)]
    df = _df(timestamps)

    gaps = quality.detect_gaps(df, "1d")

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.after_timestamp == START + 2 * DAY
    assert gap.before_timestamp == START + 7 * DAY
    assert gap.missing_bars == 4  # days 3,4,5,6


def test_expected_bar_count_matches_span():
    df = _df([START, START + 5 * DAY])
    assert quality.expected_bar_count(df, "1d") == 6  # inclusive of both ends


def test_fetch_pipeline_never_fills_a_gap_it_finds(isolated_cache_root):
    """A gap in what the exchange actually returns must survive into the
    cache untouched — no interpolation, no synthetic candle, ever."""
    candles_with_hole = [
        [START + 0 * DAY, 1.0, 1.0, 1.0, 1.0, 1.0],
        [START + 1 * DAY, 1.0, 1.0, 1.0, 1.0, 1.0],
        # day 2 and 3 are simply absent from what the exchange returns
        [START + 4 * DAY, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    exchange = FakeExchange(markets={"BTC/USDT"}, candles_by_symbol={"BTC/USDT": candles_with_hole}, now_ms=START + 10 * DAY)

    ohlcv.fetch_symbol(exchange, "BTC/USDT", "1d", "binance", min_history_bars=1)

    cached = cache.read_cached("BTC/USDT", "1d", "binance")
    assert len(cached) == 3  # exactly what was fetched, nothing invented

    gaps = quality.detect_gaps(cached, "1d")
    assert len(gaps) == 1
    assert gaps[0].missing_bars == 2
