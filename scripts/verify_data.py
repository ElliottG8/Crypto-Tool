#!/usr/bin/env python3
"""
Backfill the universe, then print a two-minute-readable data quality report.

Per symbol and timeframe: row count, first/last cached timestamp, missing
bars vs. expected, and (for BTC and ETH daily) the last 5 candles.

This script does not invent data. If a fetch fails (network, missing
symbol, etc.) that is reported as a failure for that symbol/timeframe, not
silently skipped or backfilled with a placeholder.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.universe import UNIVERSE, symbol_pairs
from data import cache, ohlcv, quality
from data.timeframes import TIMEFRAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_data")

EXCHANGE_ID = "binance"


def _fmt_ts(ms: int | None) -> str:
    if ms is None:
        return "—"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def run_backfill() -> list[ohlcv.SymbolFetchResult]:
    symbols = symbol_pairs()
    print(f"Backfilling {len(symbols)} symbols x {len(TIMEFRAMES)} timeframes on {EXCHANGE_ID} ...")
    results = ohlcv.backfill_universe(symbols=symbols, timeframes=TIMEFRAMES, exchange_id=EXCHANGE_ID)
    return results


def print_report(results: list[ohlcv.SymbolFetchResult]) -> None:
    by_symbol_tf = {(r.symbol, r.timeframe): r for r in results}

    print("\n" + "=" * 100)
    print("DATA VERIFICATION REPORT")
    print(f"Exchange: {EXCHANGE_ID}   Universe date: see config/universe.py   Generated: {_fmt_ts(int(datetime.now(timezone.utc).timestamp() * 1000))}")
    print("=" * 100)

    header = f"{'symbol':<10}{'tf':<5}{'status':<20}{'rows':>8}{'first':>20}{'last':>20}{'missing':>10}"
    print(header)
    print("-" * len(header))

    failures = []
    for base in UNIVERSE:
        symbol = f"{base}/USDT"
        for tf in TIMEFRAMES:
            result = by_symbol_tf.get((symbol, tf))
            df = cache.read_cached(symbol, tf, EXCHANGE_ID)
            first_ts = int(df["timestamp"].min()) if not df.empty else None
            last_ts = int(df["timestamp"].max()) if not df.empty else None
            gaps = quality.detect_gaps(df, tf) if not df.empty else []
            missing_total = sum(g.missing_bars for g in gaps)

            status = result.status if result else "NO_RESULT"
            print(
                f"{base:<10}{tf:<5}{status:<20}{len(df):>8}{_fmt_ts(first_ts):>20}{_fmt_ts(last_ts):>20}{missing_total:>10}"
            )

            if result and result.status in ("failed", "skipped_no_market", "insufficient_history"):
                failures.append((base, tf, result))
            if gaps:
                for g in gaps:
                    print(
                        f"    gap: {g.missing_bars} bar(s) missing between "
                        f"{_fmt_ts(g.after_timestamp)} and {_fmt_ts(g.before_timestamp)}"
                    )

    print("\n" + "-" * 100)
    print(f"{len(failures)} symbol/timeframe pair(s) NOT fully OK:")
    for base, tf, result in failures:
        detail = result.error if result.error else result.status
        print(f"  - {base} {tf}: {result.status}  ({detail})")

    print("\n" + "-" * 100)
    print("Last 5 daily candles — BTC and ETH:")
    for base in ("BTC", "ETH"):
        symbol = f"{base}/USDT"
        df = cache.read_cached(symbol, "1d", EXCHANGE_ID)
        print(f"\n  {symbol}:")
        if df.empty:
            print("    NO DATA CACHED — fetch did not succeed, nothing to show.")
            continue
        tail = df.sort_values("timestamp").tail(5)
        for _, row in tail.iterrows():
            print(
                f"    {_fmt_ts(int(row['timestamp']))}  "
                f"O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} "
                f"C:{row['close']:.2f} V:{row['volume']:.2f}"
            )

    print("\n" + "=" * 100)


def main() -> int:
    results = run_backfill()
    print_report(results)
    any_ok = any(r.status == "ok" for r in results)
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
