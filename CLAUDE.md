# Crypto Screener — Project Spec

## What this is

A multi-factor technical screener over a fixed universe of ~20 liquid crypto
assets, plus — and this is the actual deliverable — a backtest that measures
whether the screener's composite score has any out-of-sample predictive power.

The point of the project is the validation, not the signal. A screener that has
been tested and found to have no edge is a finished, honest piece of work. A
screener that produces confident rankings and has never been tested is not.
Build accordingly.

## Non-negotiable rules

1. **Never fabricate or estimate data.** If an API call fails, a symbol is
   missing, or a series has gaps, fail loudly and say so. Do not interpolate,
   do not substitute a nearby value, do not fall back on a plausible number.
   Silent imputation is the single worst failure mode in this project.

2. **Do not treat correlated indicators as independent confirmation.** RSI,
   MACD, Stochastic RSI, MA slope and "momentum" are all deterministic
   functions of the same close-price series. Six of them agreeing is one
   observation measured six ways. The scoring layer must report the empirical
   correlation matrix between factors alongside any composite score, and the
   composite must never be presented as N independent confirmations.

3. **Build in layers and verify each before moving to the next.** Do not write
   the whole pipeline in one pass. Order: data → indicators → scoring →
   backtest. Each layer gets tests before the next one starts.

4. **No lookahead.** Every value computed for timestamp `t` may only use data
   available at or before `t`. This includes normalisation: do not z-score or
   min-max a factor against the full history when scoring a historical date.
   Use expanding or rolling windows only.

## Architecture

```
data/        fetch + cache raw OHLCV, funding, OI. No computation here.
indicators/  pure functions: series in, series out. No I/O, no API calls.
scoring/     combine indicators into factor scores and a composite.
backtest/    walk-forward evaluation of the composite. The deliverable.
reports/     output only. Never the source of truth for anything.
```

Keep `indicators/` free of I/O so it is trivially testable.

## Universe

Fixed list of ~20 symbols, defined in one config file, not scattered through
the code. Chosen for liquidity and data availability. Do not expand the
universe to improve backtest results — that is how survivorship bias gets in.

Record the date the universe was chosen. Note explicitly in the report that
the universe is selected with hindsight (these coins exist and are liquid
*today*), and that this biases backtest results upward.

## Data layer

- **OHLCV**: CCXT against a major spot exchange. Public endpoints, no API key.
  Fetch 1D and 4H. Weekly can be resampled from daily.
- **Funding rate + open interest**: Binance and Bybit public futures endpoints.
  Free, no key. Not all universe symbols will have perps — handle missing
  data as missing, not as zero.
- **TVL / on-chain**: DefiLlama, free, no key. Optional; only for assets where
  it is meaningful.

Cache raw responses to disk (parquet or SQLite, keyed by symbol + timeframe +
exchange). After the initial backfill, only fetch deltas. Respect rate limits
and use CCXT's built-in throttling rather than hand-rolled sleeps.

Store raw data immutably. Never overwrite a cached candle with a "corrected"
one without logging it.

## Indicator layer

Implement: EMA/SMA (20, 50, 100, 200), RSI(14), MACD(12,26,9), Bollinger
bands and bandwidth, ATR(14), OBV, realised volatility, rolling volume ratio.

**Testing requirement.** Each indicator gets a unit test that checks its output
against a small hand-verified reference series with known expected values. A
silently wrong RSI is the classic failure here: it produces numbers in the
right range, nothing errors, and every downstream result is quietly garbage.
Do not skip this step, and do not satisfy it by asserting that the output is
between 0 and 100.

Prefer writing these yourself over pulling in `pandas-ta` or `ta-lib`. The
implementations are short, and knowing exactly what your RSI does matters more
here than saving an hour.

## Scoring layer

Produce per-factor scores, then a composite. Factors should be grouped by
what they actually measure independently — roughly:

- price/trend structure
- volume and participation
- volatility state
- derivatives positioning
- on-chain / fundamental

Within-group indicators are near-duplicates of each other. Across groups there
is more genuine independence, though still not full independence, since every
group is downstream of the same price action.

Scores must be **cross-sectional** — rank within the universe on each date —
not absolute. An absolute score just measures "is crypto up right now," which
is not a screening signal.

Output the factor correlation matrix with every scoring run.

## Backtest layer

This is the part that matters.

For each historical date in the sample, compute the composite score for every
asset in the universe using only data available at that date. Then measure
forward returns at 7D and 30D horizons.

Primary metric: cross-sectional Spearman rank correlation between score and
forward return (the information coefficient), averaged over dates, with its
standard error.

Required controls:

- **Demean by the equal-weight universe return on each date.** Otherwise you
  are measuring market beta, not selection skill.
- **Effective sample size.** Crypto assets are heavily cross-correlated, so
  20 assets × N dates is nowhere near 20N independent observations. Estimate
  the effective N from the average pairwise correlation and report it. The
  honest confidence interval will be much wider than the naive one.
- **Null comparison.** Compare the realised IC against a distribution from
  shuffled scores. If the real IC sits inside that distribution, the screener
  has no demonstrated edge — report that plainly.
- **Overlapping windows.** 30D forward returns on daily dates overlap heavily.
  Use Newey-West or block bootstrap standard errors, not naive ones.

Do not tune factor weights against the backtest and then report the backtest
as evidence. If weights are fitted, hold out a period and say so.

## Reporting

The screener output can include entries, levels and stops if useful, but every
report must carry the measured IC and its confidence interval at the top. A
ranking without its validation attached should not be producible by this
codebase.

## Environment

Python 3.11+. `ccxt`, `pandas`, `numpy`, `pyarrow`, `scipy`. Add nothing else
without a reason. Use a venv or uv. Pin versions in a lockfile.

## Not in scope

Live trading, order execution, exchange API keys with trade permissions, or
anything that touches real money. This is a research and validation project.
