from data import cache, ohlcv
from data.timeframes import TIMEFRAME_MS
from tests.fakes import FakeExchange, make_daily_candles

DAY = TIMEFRAME_MS["1d"]
START = 1_600_000_000_000  # arbitrary fixed ms timestamp, day-aligned enough for these tests


def test_delta_fetch_only_requests_candles_after_last_cached_timestamp(isolated_cache_root):
    all_candles = make_daily_candles(START, 10)
    exchange = FakeExchange(markets={"BTC/USDT"}, candles_by_symbol={"BTC/USDT": all_candles}, now_ms=START + 20 * DAY)

    # Pre-seed the cache with the first 5 days, as if a prior run already fetched them.
    seeded = cache.write_cache(
        __import__("pandas").DataFrame(all_candles[:5], columns=cache.OHLCV_COLUMNS),
        "BTC/USDT",
        "1d",
        "binance",
    )
    assert seeded.exists()

    result = ohlcv.fetch_symbol(exchange, "BTC/USDT", "1d", "binance", min_history_bars=5)

    # Every fetch_ohlcv call must ask for data strictly after the last cached candle.
    last_cached_before_fetch = all_candles[4][0]
    for call in exchange.calls:
        assert call["since"] > last_cached_before_fetch

    assert result.status == "ok"
    assert result.rows_fetched == 5  # candles 5..9 are new
    assert result.total_rows == 10

    final_cache = cache.read_cached("BTC/USDT", "1d", "binance")
    assert final_cache["timestamp"].tolist() == [c[0] for c in all_candles]


def test_first_run_backfills_from_default_since_with_no_cache(isolated_cache_root):
    all_candles = make_daily_candles(START, 3)
    exchange = FakeExchange(markets={"BTC/USDT"}, candles_by_symbol={"BTC/USDT": all_candles}, now_ms=START + 5 * DAY)

    assert cache.last_cached_timestamp("BTC/USDT", "1d", "binance") is None

    result = ohlcv.fetch_symbol(exchange, "BTC/USDT", "1d", "binance")

    assert result.rows_fetched == 3
    assert exchange.calls[0]["since"] == ohlcv.DEFAULT_BACKFILL_SINCE_MS


def test_symbol_not_on_exchange_is_excluded_not_substituted(isolated_cache_root):
    exchange = FakeExchange(markets={"ETH/USDT"}, candles_by_symbol={}, now_ms=START + 5 * DAY)

    result = ohlcv.fetch_symbol(exchange, "NOTREAL/USDT", "1d", "binance")

    assert result.status == "skipped_no_market"
    assert result.rows_fetched == 0
    assert not cache.cache_path("NOTREAL/USDT", "1d", "binance").exists()


def test_insufficient_history_is_flagged(isolated_cache_root):
    all_candles = make_daily_candles(START, 3)  # far fewer than MIN_HISTORY_BARS
    exchange = FakeExchange(markets={"BTC/USDT"}, candles_by_symbol={"BTC/USDT": all_candles}, now_ms=START + 5 * DAY)

    result = ohlcv.fetch_symbol(exchange, "BTC/USDT", "1d", "binance")

    assert result.status == "insufficient_history"
    assert result.total_rows == 3


def test_in_progress_candle_is_excluded_at_fetch(isolated_cache_root):
    closed_candles = make_daily_candles(START, 5)  # days 0..4, fully closed
    partial_open_ts = START + 5 * DAY
    partial_candle = [partial_open_ts, 100.0, 101.0, 99.0, 100.5, 42.0]  # low volume: still forming
    all_candles = closed_candles + [partial_candle]

    # Only a quarter of the way through day 5's bar — it has not closed yet.
    now_mid_bar = partial_open_ts + DAY // 4
    exchange = FakeExchange(
        markets={"BTC/USDT"},
        candles_by_symbol={"BTC/USDT": all_candles},
        now_ms=now_mid_bar,
    )

    result = ohlcv.fetch_symbol(exchange, "BTC/USDT", "1d", "binance", min_history_bars=1)

    assert result.rows_fetched == 5  # the in-progress day-5 bar must not be counted or cached
    cached = cache.read_cached("BTC/USDT", "1d", "binance")
    assert cached["timestamp"].tolist() == [c[0] for c in closed_candles]
    assert partial_open_ts not in cached["timestamp"].tolist()


def test_rerun_with_no_new_data_does_not_refetch_or_duplicate(isolated_cache_root):
    all_candles = make_daily_candles(START, 40)
    exchange = FakeExchange(markets={"BTC/USDT"}, candles_by_symbol={"BTC/USDT": all_candles}, now_ms=START + 40 * DAY)

    first = ohlcv.fetch_symbol(exchange, "BTC/USDT", "1d", "binance")
    assert first.rows_fetched == 40

    second = ohlcv.fetch_symbol(exchange, "BTC/USDT", "1d", "binance")
    assert second.rows_fetched == 0
    assert second.total_rows == 40

    final_cache = cache.read_cached("BTC/USDT", "1d", "binance")
    assert len(final_cache) == 40  # no duplicates introduced
