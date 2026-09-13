import numpy as np
import pandas as pd
import pytest


def make_close_panel(dates: pd.DatetimeIndex, symbols: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for i, sym in enumerate(symbols):
        returns = rng.normal(0.0002, 0.02, len(dates))
        data[sym] = 100.0 * (i + 1) * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def small_dates():
    return pd.date_range("2024-01-01", periods=20, freq="D")
