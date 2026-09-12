"""
Parquet-backed cache for raw OHLCV candles.

Pure persistence: read, write, and merge exactly what the exchange returned.
No indicators, no resampling, no imputation. Cached data is treated as
immutable — merging never silently overwrites a candle that's already on
disk with a different value; conflicts are reported, not resolved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parent.parent / "data_cache"

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

OHLCV_DTYPES: dict[str, str] = {
    "timestamp": "int64",  # exchange-native milliseconds since epoch, UTC
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def cache_path(symbol: str, timeframe: str, exchange: str) -> Path:
    return CACHE_ROOT / exchange / timeframe / f"{_safe_symbol(symbol)}.parquet"


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in OHLCV_DTYPES.items()})


def _validate_schema(df: pd.DataFrame) -> None:
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing required columns: {sorted(missing)}")


def read_cached(symbol: str, timeframe: str, exchange: str) -> pd.DataFrame:
    """Return cached candles for this key, or an empty typed frame if none exist."""
    path = cache_path(symbol, timeframe, exchange)
    if not path.exists():
        return _empty_frame()
    df = pd.read_parquet(path)
    return df.astype(OHLCV_DTYPES)


def last_cached_timestamp(symbol: str, timeframe: str, exchange: str) -> int | None:
    df = read_cached(symbol, timeframe, exchange)
    if df.empty:
        return None
    return int(df["timestamp"].max())


def write_cache(df: pd.DataFrame, symbol: str, timeframe: str, exchange: str) -> Path:
    """Replace the cached file for this key with df, verbatim (sorted, de-duped).

    This does not merge with any existing cache — callers merge explicitly
    via merge_new_candles first so that conflicts are surfaced before
    anything is written to disk.
    """
    _validate_schema(df)
    path = cache_path(symbol, timeframe, exchange)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    out = out.astype(OHLCV_DTYPES).reset_index(drop=True)
    out.to_parquet(path, index=False)
    return path


@dataclass
class MergeResult:
    df: pd.DataFrame
    new_rows: int
    conflicting_timestamps: list[int] = field(default_factory=list)


def merge_new_candles(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    symbol: str,
    timeframe: str,
    exchange: str,
) -> MergeResult:
    """Merge freshly fetched candles onto the existing cached frame.

    Timestamps not already cached are appended. A timestamp that IS already
    cached but whose OHLCV values disagree with the fresh fetch is logged as
    a conflict and the ORIGINAL cached value is kept — we never silently
    overwrite a cached candle with a "corrected" one.
    """
    _validate_schema(existing)
    _validate_schema(new)

    new = new.astype(OHLCV_DTYPES)

    if existing.empty:
        deduped = new.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
        return MergeResult(df=deduped.reset_index(drop=True), new_rows=len(deduped))

    existing_idx = existing.set_index("timestamp")
    new_idx = new.set_index("timestamp")

    overlap = existing_idx.index.intersection(new_idx.index)
    conflicting_timestamps: list[int] = []
    if len(overlap) > 0:
        differs = (existing_idx.loc[overlap].to_numpy() != new_idx.loc[overlap].to_numpy()).any(axis=1)
        if differs.any():
            conflicting_timestamps = [int(ts) for ts in overlap[differs]]
            logger.warning(
                "%s %s %s: %d cached candle(s) disagree with freshly fetched values at %s; "
                "keeping cached values, NOT overwriting.",
                exchange,
                symbol,
                timeframe,
                len(conflicting_timestamps),
                conflicting_timestamps,
            )

    new_only = new_idx.loc[new_idx.index.difference(existing_idx.index)]
    combined = pd.concat([existing_idx, new_only]).sort_index().reset_index()
    combined = combined.astype(OHLCV_DTYPES)

    return MergeResult(df=combined, new_rows=len(new_only), conflicting_timestamps=conflicting_timestamps)
