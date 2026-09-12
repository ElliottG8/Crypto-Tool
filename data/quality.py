"""
Raw-data quality checks: gap detection only.

This module never fills a gap — it reports one. Per project rules, missing
data stays missing; nothing here interpolates or substitutes a value.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.timeframes import TIMEFRAME_MS


@dataclass
class Gap:
    after_timestamp: int
    before_timestamp: int
    missing_bars: int


def detect_gaps(df: pd.DataFrame, timeframe: str) -> list[Gap]:
    """Find gaps in a timestamp series relative to the expected bar spacing.

    A gap is any interval between two consecutive cached candles that is
    wider than one bar. Returns one Gap per such interval, in chronological
    order. Does not touch or modify df.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unknown timeframe: {timeframe!r}")
    if df.empty or len(df) < 2:
        return []

    tf_ms = TIMEFRAME_MS[timeframe]
    ts = df["timestamp"].sort_values().to_numpy()
    diffs = ts[1:] - ts[:-1]

    gaps: list[Gap] = []
    for i in (diffs > tf_ms).nonzero()[0]:
        missing = int(diffs[i] // tf_ms) - 1
        gaps.append(
            Gap(
                after_timestamp=int(ts[i]),
                before_timestamp=int(ts[i + 1]),
                missing_bars=missing,
            )
        )
    return gaps


def expected_bar_count(df: pd.DataFrame, timeframe: str) -> int:
    """Number of bars that should exist between the first and last cached
    timestamp if there were no gaps, inclusive of both endpoints."""
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unknown timeframe: {timeframe!r}")
    if df.empty:
        return 0
    tf_ms = TIMEFRAME_MS[timeframe]
    span = int(df["timestamp"].max() - df["timestamp"].min())
    return span // tf_ms + 1
