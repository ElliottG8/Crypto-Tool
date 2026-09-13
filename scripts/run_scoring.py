#!/usr/bin/env python3
"""
Glue script: load cached daily OHLCV for the universe, run the scoring
layer, print the correlation matrix and today's composite ranking.

This is wiring, not scoring logic — all the actual computation lives in
scoring/. Reads from data_cache via data/cache.py (read-only, no fetch);
run scripts/verify_data.py first if the cache is empty or stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.universe import UNIVERSE, symbol_pairs
from data import cache
from scoring.factors import MA_SLOW_WINDOW
from scoring.scorer import score_universe

EXCHANGE_ID = "binance"
TIMEFRAME = "1d"
MIN_ROWS = MA_SLOW_WINDOW + 30  # sma200 needs 200; a little headroom to get any real signal


def load_ohlcv_panel() -> dict[str, pd.DataFrame]:
    ohlcv: dict[str, pd.DataFrame] = {}
    excluded: list[str] = []

    for base in UNIVERSE:
        symbol = f"{base}/USDT"
        df = cache.read_cached(symbol, TIMEFRAME, EXCHANGE_ID)
        if len(df) < MIN_ROWS:
            excluded.append(f"{symbol} ({len(df)} rows cached, need >= {MIN_ROWS})")
            continue
        indexed = df.set_index(pd.to_datetime(df["timestamp"], unit="ms", utc=True)).drop(columns="timestamp")
        ohlcv[symbol] = indexed[["open", "high", "low", "close", "volume"]]

    if excluded:
        print("Excluded from scoring (insufficient cached history):")
        for line in excluded:
            print(f"  - {line}")
        print()

    return ohlcv


def main() -> int:
    ohlcv = load_ohlcv_panel()
    if len(ohlcv) < 2:
        print(f"Only {len(ohlcv)} symbol(s) have enough cached history to score. "
              f"Run scripts/verify_data.py to backfill first.")
        return 1

    result = score_universe(ohlcv)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print(f"Scored {len(ohlcv)} symbols: {', '.join(sorted(ohlcv))}\n")

    print("=" * 100)
    print("FACTOR CORRELATION MATRIX (pooled cross-sectional percentile scores)")
    print("=" * 100)
    print(result.correlation_matrix.round(3))

    corr = result.correlation_matrix
    pairs = [
        (abs(corr.loc[a, b]), a, b, corr.loc[a, b])
        for i, a in enumerate(corr.columns)
        for j, b in enumerate(corr.columns)
        if j > i
    ]
    pairs.sort(reverse=True)
    print("\nMost correlated factor pairs:")
    for _, a, b, val in pairs[:10]:
        print(f"  {a:25s} {b:25s} {val:+.3f}")

    print("\n" + "=" * 100)
    print(f"WEIGHTING: {result.weighting_note}")
    print("=" * 100)

    last_date = result.composite.index.max()
    print(f"\nComposite ranking, {last_date.date()} (highest percentile first):")
    latest = result.composite.loc[last_date].dropna().sort_values(ascending=False)
    for symbol, score in latest.items():
        print(f"  {symbol:12s} {score:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
