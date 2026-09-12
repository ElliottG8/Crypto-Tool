#!/usr/bin/env python3
"""
One-off repair: delete already-cached incomplete (still-open) candles.

Before data/ohlcv.py filtered out unclosed candles at fetch time, a
partially-formed bar (period not yet closed) could get cached. Because
merge_new_candles treats cached candles as immutable and keeps the
first-cached value on any conflict, that partial bar would then be kept
forever in place of the real, closed bar a later run would otherwise fetch.

This script scans every parquet file under data_cache/, finds candles whose
period had not closed as of "now", deletes them, and reports what it
removed. It does not fetch anything or fill the gap it creates — the next
`scripts/verify_data.py` run will fetch the real closed bar to replace it.

Safe to run repeatedly; a no-op once the cache is clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data.cache import CACHE_ROOT, OHLCV_DTYPES, write_cache
from data.timeframes import TIMEFRAME_MS


def repair_all(now_ms: int | None = None) -> int:
    """Delete incomplete bars from every cached file. Returns rows removed."""
    if now_ms is None:
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    if not CACHE_ROOT.exists():
        print(f"No cache directory at {CACHE_ROOT}, nothing to repair.")
        return 0

    total_removed = 0
    for exchange_dir in sorted(p for p in CACHE_ROOT.iterdir() if p.is_dir()):
        exchange = exchange_dir.name
        for timeframe_dir in sorted(p for p in exchange_dir.iterdir() if p.is_dir()):
            timeframe = timeframe_dir.name
            tf_ms = TIMEFRAME_MS.get(timeframe)
            if tf_ms is None:
                print(f"  skipping {exchange}/{timeframe}: unrecognised timeframe")
                continue

            for path in sorted(timeframe_dir.glob("*.parquet")):
                symbol = path.stem.replace("_", "/")
                df = pd.read_parquet(path).astype(OHLCV_DTYPES)

                unclosed_mask = (df["timestamp"] + tf_ms) > now_ms
                if not unclosed_mask.any():
                    continue

                removed = df.loc[unclosed_mask]
                cleaned = df.loc[~unclosed_mask]
                write_cache(cleaned, symbol, timeframe, exchange)

                for _, row in removed.iterrows():
                    print(
                        f"  removed incomplete bar: {exchange} {symbol} {timeframe} "
                        f"open_time={int(row['timestamp'])} volume={row['volume']}"
                    )
                total_removed += len(removed)

    print(f"\nDone. Removed {total_removed} incomplete bar(s) from the cache.")
    return total_removed


if __name__ == "__main__":
    repair_all()
