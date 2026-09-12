"""
Fetch OHLCV candles via CCXT and persist them through data/cache.py.

Talks to the network (through a ccxt exchange instance) and to the parquet
cache. Does no computation — no indicators, no resampling beyond whatever
timeframe CCXT is asked for. Any symbol that fails to fetch, isn't listed on
the exchange, or comes back with too little history is logged and excluded
— never substituted from another exchange, never filled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import ccxt
import pandas as pd

from data import cache
from data.timeframes import TIMEFRAME_MS, TIMEFRAMES

logger = logging.getLogger(__name__)

# Earliest date to backfill from when there is no cache yet. If this predates
# a symbol's listing, the exchange simply returns its own earliest candles.
DEFAULT_BACKFILL_SINCE_MS = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000)

FETCH_LIMIT = 1000  # candles requested per call; exchange may cap it lower

MIN_HISTORY_BARS = 30  # below this, a symbol is "insufficient history"


@dataclass
class SymbolFetchResult:
    symbol: str
    timeframe: str
    exchange: str
    status: str  # "ok" | "skipped_no_market" | "failed" | "insufficient_history"
    rows_fetched: int = 0
    total_rows: int = 0
    error: str | None = None
    conflicting_timestamps: list[int] = field(default_factory=list)


def make_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """Build a public (no API key) CCXT exchange instance with CCXT's own
    rate limiting enabled — no hand-rolled sleeps anywhere in this module."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def _paginate_fetch(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int) -> list[list]:
    """Page forward through fetch_ohlcv from since_ms up to "now".

    Relies on the exchange instance's own throttling (enableRateLimit=True
    makes every call sleep as needed) rather than a manual sleep here.
    """
    tf_ms = TIMEFRAME_MS[timeframe]
    all_candles: list[list] = []
    cursor = since_ms
    now_ms = exchange.milliseconds()

    while cursor < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=FETCH_LIMIT)
        if not batch:
            break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            # exchange didn't advance the cursor; stop rather than loop forever
            break
        cursor = last_ts + tf_ms
        if len(batch) < FETCH_LIMIT:
            break

    return all_candles


def fetch_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    exchange_id: str,
    min_history_bars: int = MIN_HISTORY_BARS,
) -> SymbolFetchResult:
    """Fetch and cache new candles for one symbol/timeframe, incrementally.

    Only requests candles after the last cached timestamp (or from
    DEFAULT_BACKFILL_SINCE_MS if nothing is cached yet) — a rerun never
    re-downloads history that's already on disk.
    """
    markets = exchange.markets or exchange.load_markets()
    if symbol not in markets:
        logger.warning("%s %s: not listed on %s, excluding.", symbol, timeframe, exchange_id)
        return SymbolFetchResult(symbol, timeframe, exchange_id, status="skipped_no_market")

    last_ts = cache.last_cached_timestamp(symbol, timeframe, exchange_id)
    since_ms = (last_ts + TIMEFRAME_MS[timeframe]) if last_ts is not None else DEFAULT_BACKFILL_SINCE_MS

    try:
        raw = _paginate_fetch(exchange, symbol, timeframe, since_ms)
    except ccxt.BaseError as exc:
        logger.error("%s %s: fetch failed on %s: %s", symbol, timeframe, exchange_id, exc)
        return SymbolFetchResult(symbol, timeframe, exchange_id, status="failed", error=str(exc))

    existing_df = cache.read_cached(symbol, timeframe, exchange_id)

    if not raw:
        total_rows = len(existing_df)
        status = "ok" if total_rows >= min_history_bars else "insufficient_history"
        return SymbolFetchResult(symbol, timeframe, exchange_id, status=status, total_rows=total_rows)

    new_df = pd.DataFrame(raw, columns=cache.OHLCV_COLUMNS)
    merge_result = cache.merge_new_candles(existing_df, new_df, symbol, timeframe, exchange_id)
    cache.write_cache(merge_result.df, symbol, timeframe, exchange_id)

    total_rows = len(merge_result.df)
    status = "ok" if total_rows >= min_history_bars else "insufficient_history"
    return SymbolFetchResult(
        symbol,
        timeframe,
        exchange_id,
        status=status,
        rows_fetched=merge_result.new_rows,
        total_rows=total_rows,
        conflicting_timestamps=merge_result.conflicting_timestamps,
    )


def backfill_universe(
    symbols: list[str],
    timeframes: tuple[str, ...] = TIMEFRAMES,
    exchange_id: str = "binance",
    exchange: ccxt.Exchange | None = None,
) -> list[SymbolFetchResult]:
    """Incrementally backfill every symbol/timeframe pair in the universe.

    Failures on individual symbols are logged and recorded in the returned
    results, not raised — one bad symbol should not abort the whole run.
    """
    exchange = exchange or make_exchange(exchange_id)

    try:
        exchange.load_markets()
    except ccxt.BaseError as exc:
        logger.error("%s: could not load markets, aborting backfill: %s", exchange_id, exc)
        return [
            SymbolFetchResult(symbol, timeframe, exchange_id, status="failed", error=str(exc))
            for symbol in symbols
            for timeframe in timeframes
        ]

    results: list[SymbolFetchResult] = []
    for symbol in symbols:
        for timeframe in timeframes:
            results.append(fetch_symbol(exchange, symbol, timeframe, exchange_id))
    return results
