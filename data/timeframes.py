"""Shared timeframe constants for the data layer."""

TIMEFRAMES: tuple[str, ...] = ("1d", "4h")

TIMEFRAME_MS: dict[str, int] = {
    "1d": 24 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}
