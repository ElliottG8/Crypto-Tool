import pandas as pd

from data import cache


def _sample_df(timestamps):
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(i) + 0.1 for i in range(len(timestamps))],
            "high": [float(i) + 0.2 for i in range(len(timestamps))],
            "low": [float(i) - 0.1 for i in range(len(timestamps))],
            "close": [float(i) + 0.05 for i in range(len(timestamps))],
            "volume": [1000.0 + i for i in range(len(timestamps))],
        }
    )


def test_round_trip_preserves_values_and_dtypes(isolated_cache_root):
    df = _sample_df([1_000, 2_000, 3_000])

    cache.write_cache(df, "BTC/USDT", "1d", "binance")
    read_back = cache.read_cached("BTC/USDT", "1d", "binance")

    pd.testing.assert_frame_equal(
        read_back.reset_index(drop=True),
        df.astype(cache.OHLCV_DTYPES).reset_index(drop=True),
    )
    for col, dtype in cache.OHLCV_DTYPES.items():
        assert str(read_back[col].dtype) == dtype


def test_read_cached_missing_key_returns_empty_typed_frame(isolated_cache_root):
    result = cache.read_cached("NOPE/USDT", "1d", "binance")
    assert result.empty
    for col, dtype in cache.OHLCV_DTYPES.items():
        assert str(result[col].dtype) == dtype


def test_last_cached_timestamp(isolated_cache_root):
    assert cache.last_cached_timestamp("BTC/USDT", "1d", "binance") is None

    cache.write_cache(_sample_df([1_000, 5_000, 3_000]), "BTC/USDT", "1d", "binance")
    assert cache.last_cached_timestamp("BTC/USDT", "1d", "binance") == 5_000


def test_merge_appends_only_new_timestamps(isolated_cache_root):
    existing = _sample_df([1_000, 2_000])
    # Build `new` so the overlapping 2_000 row is byte-identical to `existing`'s.
    new = pd.concat([existing.iloc[[1]], _sample_df([3_000, 4_000])], ignore_index=True)

    result = cache.merge_new_candles(existing, new, "BTC/USDT", "1d", "binance")

    assert sorted(result.df["timestamp"].tolist()) == [1_000, 2_000, 3_000, 4_000]
    assert result.new_rows == 2  # only 3_000 and 4_000 are genuinely new
    assert result.conflicting_timestamps == []


def test_merge_never_overwrites_conflicting_cached_candle(isolated_cache_root):
    existing = _sample_df([1_000, 2_000])
    corrupted_new = _sample_df([2_000, 3_000])
    corrupted_new.loc[corrupted_new["timestamp"] == 2_000, "close"] = 999_999.0

    result = cache.merge_new_candles(existing, corrupted_new, "BTC/USDT", "1d", "binance")

    kept = result.df.loc[result.df["timestamp"] == 2_000, "close"].iloc[0]
    original = existing.loc[existing["timestamp"] == 2_000, "close"].iloc[0]
    assert kept == original  # cached value wins, never silently overwritten
    assert result.conflicting_timestamps == [2_000]


def test_write_cache_deduplicates_and_sorts(isolated_cache_root):
    unsorted_with_dupe = _sample_df([3_000, 1_000, 3_000])

    cache.write_cache(unsorted_with_dupe, "BTC/USDT", "1d", "binance")
    read_back = cache.read_cached("BTC/USDT", "1d", "binance")

    assert read_back["timestamp"].tolist() == [1_000, 3_000]
