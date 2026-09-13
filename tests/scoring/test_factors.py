import numpy as np
import pandas as pd
import pytest

from scoring.factors import atr_pct, compute_all_factors, ma_pct_distance, macd_histogram_pct, obv_participation
from tests.scoring.conftest import make_ohlcv


def _scale_price_columns(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    scaled = df.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] = scaled[col] * factor
    return scaled  # volume untouched


@pytest.mark.parametrize("factor", [0.001, 1000.0])
def test_macd_histogram_pct_is_scale_invariant(factor):
    df = make_ohlcv(120, seed=1, base_price=100.0)
    scaled = _scale_price_columns(df, factor)
    original = macd_histogram_pct(df["close"])
    rescaled = macd_histogram_pct(scaled["close"])
    pd.testing.assert_series_equal(original, rescaled, check_names=False, rtol=1e-9)


@pytest.mark.parametrize("factor", [0.001, 1000.0])
def test_ma_pct_distance_is_scale_invariant(factor):
    df = make_ohlcv(120, seed=1, base_price=100.0)
    scaled = _scale_price_columns(df, factor)
    original = ma_pct_distance(df["close"], window=50)
    rescaled = ma_pct_distance(scaled["close"], window=50)
    pd.testing.assert_series_equal(original, rescaled, check_names=False, rtol=1e-9)


@pytest.mark.parametrize("factor", [0.001, 1000.0])
def test_atr_pct_is_scale_invariant(factor):
    df = make_ohlcv(60, seed=1, base_price=100.0)
    scaled = _scale_price_columns(df, factor)
    original = atr_pct(df["high"], df["low"], df["close"], window=14)
    rescaled = atr_pct(scaled["high"], scaled["low"], scaled["close"], window=14)
    pd.testing.assert_series_equal(original, rescaled, check_names=False, rtol=1e-9)


@pytest.mark.parametrize("factor", [0.001, 1000.0])
def test_obv_participation_is_scale_invariant(factor):
    # OBV only uses the DIRECTION of price change (and volume), so scaling
    # price by any positive constant must not change it at all.
    df = make_ohlcv(60, seed=1, base_price=100.0)
    scaled = _scale_price_columns(df, factor)
    original = obv_participation(df["close"], df["volume"], window=20)
    rescaled = obv_participation(scaled["close"], scaled["volume"], window=20)
    pd.testing.assert_series_equal(original, rescaled, check_names=False, rtol=1e-9)


@pytest.mark.parametrize("factor", [0.001, 1000.0])
def test_every_factor_is_scale_invariant_under_price_scaling(factor):
    """compute_all_factors output must be identical whether an asset trades
    at $0.0001 or $1,000,000 — that's the whole point of converting every
    price-unit indicator into a ratio before it enters a score."""
    df = make_ohlcv(220, seed=4, base_price=100.0)
    scaled = _scale_price_columns(df, factor)

    original = compute_all_factors(df)
    rescaled = compute_all_factors(scaled)

    assert original.keys() == rescaled.keys()
    for name in original:
        pd.testing.assert_series_equal(
            original[name], rescaled[name], check_names=False, rtol=1e-9,
            obj=f"factor {name!r} under price scale x{factor}",
        )
