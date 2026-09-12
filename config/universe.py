"""
Fixed screening universe.

*** EDITABLE — this is the one place the universe is defined. ***
Do not hardcode symbol lists anywhere else in the codebase; import from here.

Chosen for liquidity and data availability as of the date below. Do NOT
expand this list to improve backtest results after the fact — that is
survivorship bias by another name. Any change to this list should be a
deliberate, logged decision, not a tuning knob.

NOTE (see CLAUDE.md, "Universe"): these symbols are liquid and actively
traded *today*. Selecting them with the benefit of hindsight biases any
backtest run over their full history upward — a symbol that had gone to
zero or been delisted by now would not appear on this list. This bias is
structural and cannot be fixed after the fact; it can only be disclosed.
Report this plainly alongside any backtest results.
"""

from datetime import date

# Date this universe list was chosen / last deliberately edited.
UNIVERSE_SELECTED_DATE = date(2026, 9, 12)

# Base assets, quoted against USDT on the exchange configured in
# data/ohlcv.py. Editable — but see the warning above.
UNIVERSE: list[str] = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "BNB",
    "ADA",
    "AVAX",
    "LINK",
    "DOT",
    "MATIC",
    "LTC",
    "ATOM",
    "UNI",
    "AAVE",
    "HBAR",
    "NEAR",
    "ARB",
    "OP",
    "INJ",
    "FLR",
]

QUOTE = "USDT"


def symbol_pairs() -> list[str]:
    """Return CCXT-style 'BASE/QUOTE' trading pairs for the universe."""
    return [f"{base}/{QUOTE}" for base in UNIVERSE]
