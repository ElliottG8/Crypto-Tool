import pandas as pd
import pytest

from scoring.scorer import score_universe
from tests.scoring.conftest import make_ohlcv


def test_score_universe_no_lookahead(synthetic_universe):
    """Truncating every symbol's history to its first `cut` dates must not
    change any already-computed factor score, group score, or composite
    for those dates — the whole pipeline (per-asset factors, cross-
    sectional ranking, group/composite averaging) is causal end to end."""
    full_result = score_universe(synthetic_universe)

    cut = 150
    truncated_universe = {symbol: df.iloc[:cut] for symbol, df in synthetic_universe.items()}
    truncated_result = score_universe(truncated_universe)

    for name, full_panel in full_result.factor_scores.items():
        truncated_panel = truncated_result.factor_scores[name]
        common_index = truncated_panel.index
        pd.testing.assert_frame_equal(
            truncated_panel,
            full_panel.loc[common_index],
            check_names=False,
            obj=f"factor_scores[{name!r}]",
        )

    for name, full_panel in full_result.group_scores.items():
        truncated_panel = truncated_result.group_scores[name]
        pd.testing.assert_frame_equal(
            truncated_panel,
            full_panel.loc[truncated_panel.index],
            check_names=False,
            obj=f"group_scores[{name!r}]",
        )

    common_index = truncated_result.composite.index
    pd.testing.assert_frame_equal(
        truncated_result.composite,
        full_result.composite.loc[common_index],
        check_names=False,
    )


def test_score_universe_is_scale_invariant_to_one_assets_price_level(synthetic_universe):
    """Scaling one asset's entire price series by a constant must not
    change ANY factor score, group score, or composite for ANY asset —
    not just the rescaled one. Since every price-unit indicator is
    converted to a ratio before scoring, the rescaled asset's own factor
    values are bit-for-bit unchanged, so the whole cross-section (which
    only depends on relative values) is unchanged too."""
    scaled_universe = {symbol: df.copy() for symbol, df in synthetic_universe.items()}
    for col in ("open", "high", "low", "close"):
        scaled_universe["BTC/USDT"][col] = scaled_universe["BTC/USDT"][col] * 1000.0

    original_result = score_universe(synthetic_universe)
    scaled_result = score_universe(scaled_universe)

    for name in original_result.factor_scores:
        pd.testing.assert_frame_equal(
            original_result.factor_scores[name],
            scaled_result.factor_scores[name],
            check_names=False,
            rtol=1e-9,
            obj=f"factor_scores[{name!r}]",
        )

    pd.testing.assert_frame_equal(
        original_result.composite, scaled_result.composite, check_names=False, rtol=1e-9
    )


def test_score_universe_reports_correlation_matrix_alongside_composite(synthetic_universe):
    result = score_universe(synthetic_universe)
    assert result.correlation_matrix is not None
    assert set(result.correlation_matrix.columns) == set(result.factor_scores.keys())
    assert result.composite is not None


def test_score_universe_documents_unfitted_equal_weighting(synthetic_universe):
    result = score_universe(synthetic_universe)
    assert "unfitted" in result.weighting_note.lower() or "not fitted" in result.weighting_note.lower()


def test_score_universe_rejects_empty_input():
    with pytest.raises(ValueError):
        score_universe({})
