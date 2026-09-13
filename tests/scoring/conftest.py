import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int, seed: int, base_price: float, start: str = "2024-01-01") -> pd.DataFrame:
    """Deterministic synthetic OHLCV via a small-step geometric random walk
    (never goes negative or zero, unlike an additive walk)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    returns = rng.normal(0, 0.02, n)
    close = pd.Series(base_price * np.cumprod(1 + returns), index=dates)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.uniform(1_000, 5_000, n), index=dates)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


@pytest.fixture
def synthetic_universe() -> dict[str, pd.DataFrame]:
    return {
        "BTC/USDT": make_ohlcv(250, seed=1, base_price=50_000.0),
        "ETH/USDT": make_ohlcv(250, seed=2, base_price=3_000.0),
        "ADA/USDT": make_ohlcv(230, seed=3, base_price=0.5),  # shorter history, later listing
    }
