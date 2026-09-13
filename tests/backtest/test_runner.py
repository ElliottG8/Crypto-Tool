import numpy as np
import pandas as pd
import pytest

from backtest.runner import run_backtest
from scoring.scorer import score_universe


def _make_ohlcv(n: int, seed: int, base_price: float, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    returns = rng.normal(0.0002, 0.02, n)
    close = pd.Series(base_price * np.cumprod(1 + returns), index=dates)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.uniform(1_000, 5_000, n), index=dates)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


@pytest.fixture
def small_universe():
    return {
        "AAA/USDT": _make_ohlcv(260, seed=1, base_price=100.0),
        "BBB/USDT": _make_ohlcv(260, seed=2, base_price=50.0),
        "CCC/USDT": _make_ohlcv(260, seed=3, base_price=10.0),
        "DDD/USDT": _make_ohlcv(260, seed=4, base_price=5.0),
        "EEE/USDT": _make_ohlcv(260, seed=5, base_price=2.0),
        "FFF/USDT": _make_ohlcv(260, seed=6, base_price=1.0),
    }


def test_run_backtest_reports_composite_and_every_group(small_universe):
    sresult = score_universe(small_universe)
    bresult = run_backtest(small_universe, sresult, horizons=(7, 30), n_shuffles=20)

    for horizon in (7, 30):
        hr = bresult.horizons[horizon]
        assert set(hr.ic_by_score.keys()) == {"composite", "price_trend", "volume_participation", "volatility_state"}
        assert set(hr.null_by_score.keys()) == set(hr.ic_by_score.keys())
        for name, ic_result in hr.ic_by_score.items():
            assert ic_result.horizon == horizon
            assert ic_result.name == name


def test_run_backtest_reports_effective_n_and_avg_correlation_per_horizon(small_universe):
    sresult = score_universe(small_universe)
    bresult = run_backtest(small_universe, sresult, horizons=(7, 30), n_shuffles=10)

    for horizon in (7, 30):
        hr = bresult.horizons[horizon]
        assert hr.nominal_n_assets == 6
        assert 1.0 <= hr.effective_n_assets <= 6.0
        assert -1.0 <= hr.avg_pairwise_correlation <= 1.0


def test_run_backtest_assets_per_date_changes_over_time_with_staggered_listings():
    """POL-style staggered listing: an asset with no history before some
    date must show up in the cross-section count only once it's there —
    and only once it ALSO has enough of its own history for every factor
    (the longest lookback here is sma200) — the count must not be constant
    across the whole sample."""
    universe = {
        "AAA/USDT": _make_ohlcv(600, seed=1, base_price=100.0, start="2023-01-01"),
        "BBB/USDT": _make_ohlcv(600, seed=2, base_price=50.0, start="2023-01-01"),
        "CCC/USDT": _make_ohlcv(600, seed=3, base_price=10.0, start="2023-01-01"),
        "LATE/USDT": _make_ohlcv(260, seed=4, base_price=1.0, start="2023-10-01"),  # lists much later
    }
    sresult = score_universe(universe)
    bresult = run_backtest(universe, sresult, horizons=(7,), n_shuffles=5)

    counts = bresult.assets_per_date
    assert counts.nunique() > 1  # the count DOES change over time

    early_count = counts.loc[counts.index < "2023-10-01"].max()
    late_count = counts.loc[counts.index >= "2024-05-01"].max()  # well after LATE's own 200-day warmup
    assert early_count == 3  # only AAA, BBB, CCC exist yet
    assert late_count == 4  # LATE/USDT has since joined the cross-section


def test_run_backtest_rejects_empty_ohlcv():
    class FakeScoringResult:
        composite = pd.DataFrame()
        group_scores = {}

    with pytest.raises(ValueError):
        run_backtest({}, FakeScoringResult())
