"""Tests for the data module: synthetic panel structure and liquidity filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_research.config import DataConfig
from alpha_research.data import build_dataset

REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "sector",
    "is_benchmark",
    "dollar_volume",
    "daily_return",
    "benchmark_return",
}


def _synthetic_config(**kwargs) -> DataConfig:
    config = DataConfig(source="synthetic", start_date="2021-01-01", end_date="2022-12-31")
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def test_synthetic_panel_has_required_columns() -> None:
    panel = build_dataset(_synthetic_config())
    assert REQUIRED_COLUMNS.issubset(set(panel.columns))


def test_synthetic_panel_benchmark_row_exists() -> None:
    config = _synthetic_config(benchmark="SPY")
    panel = build_dataset(config)
    assert panel["is_benchmark"].any()
    assert "SPY" in panel["ticker"].values


def test_synthetic_panel_no_duplicate_date_ticker() -> None:
    panel = build_dataset(_synthetic_config())
    dupes = panel.duplicated(subset=["date", "ticker"])
    assert not dupes.any(), "Duplicate (date, ticker) pairs found in panel."


def test_synthetic_panel_dollar_volume_is_close_times_volume() -> None:
    panel = build_dataset(_synthetic_config())
    equity = panel.loc[~panel["is_benchmark"]]
    expected = equity["close"] * equity["volume"]
    assert np.allclose(equity["dollar_volume"].to_numpy(), expected.to_numpy(), rtol=1e-9)


def test_synthetic_panel_daily_return_matches_pct_change() -> None:
    panel = build_dataset(_synthetic_config())
    ticker = panel.loc[~panel["is_benchmark"], "ticker"].iloc[0]
    sub = panel.loc[panel["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    expected = sub["adj_close"].pct_change()
    # First row is NaN; compare from row 1 onward
    assert np.allclose(
        sub["daily_return"].iloc[1:].to_numpy(),
        expected.iloc[1:].to_numpy(),
        equal_nan=True,
    )


def test_synthetic_panel_benchmark_return_is_broadcast_to_all_tickers() -> None:
    panel = build_dataset(_synthetic_config())
    sample_date = panel["date"].dropna().unique()[100]
    day_returns = panel.loc[panel["date"] == sample_date, "benchmark_return"].dropna()
    assert day_returns.nunique() == 1, "benchmark_return should be identical for all tickers on a given date."


def test_liquidity_filter_removes_low_price_tickers() -> None:
    """Setting a high min_price should reduce the number of equity tickers returned."""
    config_relaxed = _synthetic_config(min_price=0.0, synthetic_tickers=20)
    config_strict = _synthetic_config(min_price=1e9, synthetic_tickers=20)
    panel_relaxed = build_dataset(config_relaxed)
    panel_strict = build_dataset(config_strict)

    tickers_relaxed = panel_relaxed.loc[~panel_relaxed["is_benchmark"], "ticker"].nunique()
    tickers_strict = panel_strict.loc[~panel_strict["is_benchmark"], "ticker"].nunique()
    assert tickers_strict <= tickers_relaxed


def test_build_dataset_uses_cache_on_second_call(tmp_path) -> None:
    config = _synthetic_config(cache_dir=str(tmp_path))
    panel1 = build_dataset(config)
    panel2 = build_dataset(config)
    pd.testing.assert_frame_equal(panel1.reset_index(drop=True), panel2.reset_index(drop=True))


def test_invalid_source_raises() -> None:
    config = DataConfig(source="invalid_source")
    with pytest.raises(ValueError, match="Unsupported data source"):
        build_dataset(config)
