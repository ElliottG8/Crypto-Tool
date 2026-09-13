"""
Walk-forward evaluation of the composite (and each group) score against
forward returns. The deliverable, per CLAUDE.md: this is what determines
whether the screener has any demonstrated edge. No I/O, no weight fitting
— consumes what scoring/ already produced and measures it.
"""

from backtest.effective_sample_size import average_pairwise_correlation, effective_n_assets
from backtest.ic import ICResult, daily_spearman_ic, summarize_ic
from backtest.newey_west import newey_west_se
from backtest.null_distribution import NullComparison, compare_to_null, shuffled_null_mean_ics
from backtest.returns import demean_by_universe, forward_returns
from backtest.runner import BacktestResult, HorizonResult, run_backtest

__all__ = [
    "forward_returns",
    "demean_by_universe",
    "newey_west_se",
    "daily_spearman_ic",
    "summarize_ic",
    "ICResult",
    "average_pairwise_correlation",
    "effective_n_assets",
    "shuffled_null_mean_ics",
    "compare_to_null",
    "NullComparison",
    "run_backtest",
    "BacktestResult",
    "HorizonResult",
]
