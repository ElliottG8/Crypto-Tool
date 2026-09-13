"""
Top-level backtest orchestration: OHLCV + a ScoringResult in, a
BacktestResult out. Walk-forward evaluation only — no I/O, no weight
fitting. Nothing here feeds back into scoring/: this layer only measures
what scoring/ already produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtest.effective_sample_size import average_pairwise_correlation, effective_n_assets
from backtest.ic import ICResult, daily_spearman_ic, summarize_ic
from backtest.null_distribution import NullComparison, compare_to_null, shuffled_null_mean_ics
from backtest.returns import demean_by_universe, forward_returns
from scoring.cross_section import build_panel

DEFAULT_HORIZONS: tuple[int, ...] = (7, 30)
DEFAULT_N_SHUFFLES = 200
DEFAULT_MIN_ASSETS = 5

METHODOLOGY_NOTES = (
    "Forward returns are demeaned by the equal-weight universe return on "
    "each date (backtest/returns.py), so IC measures cross-sectional "
    "selection skill, not market beta. "
    "Standard errors use Newey-West (Bartlett kernel) with lag = horizon "
    "length, since a horizon-D forward return computed on consecutive "
    "daily dates overlaps for D-1 of its D days. "
    "Effective sample size is estimated from the average pairwise "
    "correlation of asset returns (backtest/effective_sample_size.py) and "
    "reported alongside the nominal date/asset counts — it is not folded "
    "into the Newey-West interval, which corrects for time-series "
    "autocorrelation only; both push the honest uncertainty wider than "
    "either shown alone. "
    "The null distribution comes from shuffling scores within each date "
    "(preserving the cross-sectional structure of both the score "
    "distribution and the real, correlated returns) and recomputing mean "
    "IC many times. "
    "No factor weight, factor selection, or horizon was changed based on "
    "this result. This layer reports IC only — no Sharpe ratio, equity "
    "curve, or simulated PnL, which would imply a tradeable strategy that "
    "has not been established."
)


@dataclass
class HorizonResult:
    horizon: int
    ic_by_score: dict[str, ICResult]
    null_by_score: dict[str, NullComparison]
    avg_pairwise_correlation: float
    effective_n_assets: float
    nominal_n_assets: int


@dataclass
class BacktestResult:
    horizons: dict[int, HorizonResult]
    assets_per_date: pd.Series  # composite-scoreable cross-section size, every date
    methodology_notes: str = field(default=METHODOLOGY_NOTES)


def run_backtest(
    ohlcv: dict[str, pd.DataFrame],
    scoring_result,  # scoring.scorer.ScoringResult — not type-imported to avoid a hard scoring<->backtest cycle in tests that stub it
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    min_assets: int = DEFAULT_MIN_ASSETS,
    null_seed: int = 0,
) -> BacktestResult:
    if not ohlcv:
        raise ValueError("ohlcv must not be empty")

    close_panel = build_panel({symbol: df["close"] for symbol, df in ohlcv.items()})
    assets_per_date = scoring_result.composite.notna().sum(axis=1)

    scores_to_test: dict[str, pd.DataFrame] = {"composite": scoring_result.composite, **scoring_result.group_scores}

    horizon_results: dict[int, HorizonResult] = {}
    for horizon in horizons:
        fwd = forward_returns(close_panel, horizon)
        demeaned = demean_by_universe(fwd)

        avg_corr = average_pairwise_correlation(fwd)
        eff_n = effective_n_assets(n_assets=close_panel.shape[1], avg_pairwise_corr=avg_corr)

        ic_by_score: dict[str, ICResult] = {}
        null_by_score: dict[str, NullComparison] = {}
        for name, score_panel in scores_to_test.items():
            daily_ic = daily_spearman_ic(score_panel, demeaned, min_assets=min_assets)
            ic_result = summarize_ic(name, horizon, daily_ic)
            ic_by_score[name] = ic_result

            null_draws = shuffled_null_mean_ics(
                score_panel, demeaned, n_shuffles=n_shuffles, min_assets=min_assets, seed=null_seed
            )
            null_by_score[name] = compare_to_null(ic_result.mean_ic, null_draws)

        horizon_results[horizon] = HorizonResult(
            horizon=horizon,
            ic_by_score=ic_by_score,
            null_by_score=null_by_score,
            avg_pairwise_correlation=avg_corr,
            effective_n_assets=eff_n,
            nominal_n_assets=close_panel.shape[1],
        )

    return BacktestResult(horizons=horizon_results, assets_per_date=assets_per_date)
