#!/usr/bin/env python3
"""
Glue script: load cached daily OHLCV, run scoring then the backtest,
print the IC report required at the top of every report per CLAUDE.md.

This is wiring, not backtest logic — all the actual computation lives in
backtest/. Reads from data_cache via data/cache.py (read-only, no fetch);
run scripts/verify_data.py first if the cache is empty or stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backtest.runner import run_backtest
from scoring.scorer import score_universe
from scripts.run_scoring import load_ohlcv_panel

N_SHUFFLES = 500


def print_horizon_report(horizon: int, hr) -> None:
    print("\n" + "=" * 100)
    print(f"HORIZON: {horizon}D")
    print("=" * 100)
    print(
        f"Nominal N assets: {hr.nominal_n_assets}   "
        f"Avg pairwise return correlation: {hr.avg_pairwise_correlation:+.3f}   "
        f"Effective N assets: {hr.effective_n_assets:.2f}"
    )
    print(
        "(honest confidence intervals below are wider than a naive one that treats every "
        "date x asset cell as independent — see backtest.runner.METHODOLOGY_NOTES)\n"
    )

    header = f"{'score':<22}{'mean IC':>10}{'NW SE':>10}{'95% CI':>24}{'n dates':>10}{'null pct':>10}{'p-value':>10}"
    print(header)
    print("-" * len(header))
    for name, ic in hr.ic_by_score.items():
        null = hr.null_by_score[name]
        ci = f"[{ic.ci_low:+.4f}, {ic.ci_high:+.4f}]"
        print(
            f"{name:<22}{ic.mean_ic:>+10.4f}{ic.newey_west_se:>10.4f}{ci:>24}"
            f"{ic.n_dates:>10}{null.percentile_of_real:>9.1f}%{null.two_sided_p_value:>10.3f}"
        )


def main() -> int:
    ohlcv = load_ohlcv_panel()
    if len(ohlcv) < 2:
        print(f"Only {len(ohlcv)} symbol(s) have enough cached history to backtest. "
              f"Run scripts/verify_data.py to backfill first.")
        return 1

    print(f"Scoring {len(ohlcv)} symbols: {', '.join(sorted(ohlcv))}")
    sresult = score_universe(ohlcv)

    print(f"Running walk-forward backtest ({N_SHUFFLES} null shuffles per horizon/score)...")
    bresult = run_backtest(ohlcv, sresult, n_shuffles=N_SHUFFLES)

    pd.set_option("display.width", 160)

    counts = bresult.assets_per_date
    print("\n" + "=" * 100)
    print("CROSS-SECTION SIZE OVER TIME (composite-scoreable assets per date)")
    print("=" * 100)
    print(f"min={counts.min()}  max={counts.max()}  distinct values over the sample: {sorted(counts.unique())}")
    if counts.nunique() <= 1:
        print("WARNING: the cross-section size never changed — check the universe/data range.")

    for horizon, hr in bresult.horizons.items():
        print_horizon_report(horizon, hr)

    print("\n" + "=" * 100)
    print("METHODOLOGY")
    print("=" * 100)
    print(bresult.methodology_notes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
