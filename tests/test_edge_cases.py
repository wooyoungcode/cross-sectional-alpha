"""Edge case and robustness tests across the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_research.backtest import run_backtest
from alpha_research.config import BacktestConfig, PortfolioConfig
from alpha_research.portfolio import construct_portfolio

# ---------------------------------------------------------------------------
# Portfolio edge cases
# ---------------------------------------------------------------------------

def test_portfolio_returns_empty_when_too_few_names() -> None:
    """construct_portfolio should return an empty DataFrame if not enough names per side."""
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "ticker": ["A", "B", "C"],
            "sector": ["Tech", "Tech", "Health"],
            "beta_60d": [1.0, 1.0, 1.0],
            "prediction": [0.9, -0.9, 0.1],
            "split": ["test"] * 3,
        }
    )
    config = PortfolioConfig(
        long_quantile=0.5, short_quantile=0.5, rebalance_frequency_days=1, min_names_per_side=5
    )
    weights = construct_portfolio(predictions, exposures=None, portfolio_config=config)
    assert weights.empty


def test_portfolio_weights_sum_to_zero() -> None:
    """Net weight of a dollar-neutral portfolio must be exactly zero."""
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10,
            "ticker": list("ABCDEFGHIJ"),
            "sector": ["Tech"] * 5 + ["Health"] * 5,
            "beta_60d": [1.0] * 10,
            "prediction": [1.0, 0.9, 0.8, 0.7, 0.6, -0.6, -0.7, -0.8, -0.9, -1.0],
            "split": ["test"] * 10,
        }
    )
    config = PortfolioConfig(
        long_quantile=0.4,
        short_quantile=0.4,
        rebalance_frequency_days=1,
        min_names_per_side=2,
        beta_neutral=False,
        sector_neutral=False,
    )
    weights = construct_portfolio(predictions, exposures=None, portfolio_config=config)
    assert np.isclose(weights["weight"].sum(), 0.0, atol=1e-8)


def test_portfolio_gross_exposure_is_one() -> None:
    """Gross exposure (sum of |weights|) should be normalised to 1."""
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 8,
            "ticker": list("ABCDEFGH"),
            "sector": ["Tech", "Tech", "Health", "Health", "Tech", "Tech", "Health", "Health"],
            "beta_60d": [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "prediction": [0.9, 0.8, 0.7, 0.6, -0.6, -0.7, -0.8, -0.9],
            "split": ["test"] * 8,
        }
    )
    config = PortfolioConfig(
        long_quantile=0.5,
        short_quantile=0.5,
        rebalance_frequency_days=1,
        min_names_per_side=2,
    )
    weights = construct_portfolio(predictions, exposures=None, portfolio_config=config)
    gross = weights["weight"].abs().sum()
    assert np.isclose(gross, 1.0, atol=1e-8)


def test_portfolio_no_nan_weights() -> None:
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 8,
            "ticker": list("ABCDEFGH"),
            "sector": ["Tech"] * 4 + ["Health"] * 4,
            "beta_60d": [1.0] * 8,
            "prediction": [0.9, 0.8, 0.7, 0.6, -0.6, -0.7, -0.8, -0.9],
            "split": ["test"] * 8,
        }
    )
    config = PortfolioConfig(
        long_quantile=0.5, short_quantile=0.5, rebalance_frequency_days=1, min_names_per_side=2
    )
    weights = construct_portfolio(predictions, exposures=None, portfolio_config=config)
    assert not weights["weight"].isna().any()


# ---------------------------------------------------------------------------
# Backtest edge cases
# ---------------------------------------------------------------------------

def test_backtest_raises_on_empty_weights() -> None:
    weights = pd.DataFrame(columns=["date", "ticker", "weight", "prediction", "sector", "beta_60d", "split"])
    returns = pd.DataFrame({"date": [], "ticker": [], "daily_return": [], "benchmark_return": []})
    with pytest.raises(ValueError, match="No portfolio weights"):
        run_backtest(weights, returns, BacktestConfig())


def test_backtest_net_return_lte_gross_return_always() -> None:
    """Net return can only be less than or equal to gross due to transaction costs."""
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "ticker": ["A", "B"],
            "weight": [0.5, -0.5],
            "prediction": [1.0, -1.0],
            "sector": ["Tech", "Tech"],
            "beta_60d": [1.0, 1.0],
            "split": ["test", "test"],
        }
    )
    returns = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-03"),
            ],
            "ticker": ["A", "B", "A", "B"],
            "daily_return": [0.02, -0.02, 0.02, -0.02],
            "benchmark_return": [0.01, 0.01, 0.01, 0.01],
        }
    )
    result = run_backtest(weights, returns, BacktestConfig(transaction_cost_bps=50.0))
    assert not result.daily_results.empty
    assert (result.daily_results["net_return"] <= result.daily_results["gross_return"] + 1e-12).all()


def test_backtest_cumulative_return_starts_near_zero() -> None:
    """On the first day cumulative return should equal net return."""
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "ticker": ["A", "B"],
            "weight": [0.5, -0.5],
            "prediction": [1.0, -1.0],
            "sector": ["Tech", "Tech"],
            "beta_60d": [1.0, 1.0],
            "split": ["test", "test"],
        }
    )
    returns = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-03"),
            ],
            "ticker": ["A", "B", "A", "B"],
            "daily_return": [0.01, -0.01, 0.01, -0.01],
            "benchmark_return": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = run_backtest(weights, returns, BacktestConfig())
    assert not result.daily_results.empty
    first_row = result.daily_results.iloc[0]
    assert np.isclose(first_row["cumulative_return"], first_row["net_return"], atol=1e-10)


def test_backtest_drawdown_is_non_positive() -> None:
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "ticker": ["A", "B"],
            "weight": [0.5, -0.5],
            "prediction": [1.0, -1.0],
            "sector": ["Tech", "Tech"],
            "beta_60d": [1.0, 1.0],
            "split": ["test", "test"],
        }
    )
    returns = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-03"),
            ],
            "ticker": ["A", "B", "A", "B"],
            "daily_return": [0.01, -0.01, 0.01, -0.01],
            "benchmark_return": [0.01, 0.01, 0.01, 0.01],
        }
    )
    result = run_backtest(weights, returns, BacktestConfig())
    assert not result.daily_results.empty
    assert (result.daily_results["drawdown"] <= 0.0 + 1e-12).all()
