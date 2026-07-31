"""Backtesting engine: return attribution, turnover, drawdown, and regime analysis.

Expands rebalance-date weights to daily holdings, computes gross and net
(after transaction costs) P&L, and summarises performance via annualised
Sharpe, max drawdown, hit rate, and volatility-regime breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass(slots=True)
class BacktestResult:
    daily_results: pd.DataFrame
    metrics: pd.DataFrame
    leg_contributions: pd.DataFrame
    regime_metrics: pd.DataFrame
    weights: pd.DataFrame


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_config: BacktestConfig,
) -> BacktestResult:
    """Run a daily long/short backtest and return a structured result.

    Weights are held constant between rebalance dates (next-day execution
    assumed). Transaction costs are applied as a flat bps charge on daily
    turnover.

    Parameters
    ----------
    weights:
        Portfolio weights as returned by
        :func:`~alpha_research.portfolio.construct_portfolio`. Must contain
        ``date``, ``ticker``, and ``weight``.
    returns:
        Daily returns panel containing ``date``, ``ticker``, ``daily_return``,
        and the benchmark return column specified in ``cost_config``.
    cost_config:
        Transaction cost rate, annualisation factor, benchmark column name,
        and regime volatility window.

    Returns
    -------
    BacktestResult
        Dataclass with ``daily_results``, ``metrics``, ``leg_contributions``,
        ``regime_metrics``, and ``weights``.

    Raises
    ------
    ValueError
        If ``weights`` is empty (no valid rebalance dates were produced).
    """
    returns_frame = returns[["date", "ticker", "daily_return", cost_config.benchmark_column]].copy()
    returns_frame["date"] = pd.to_datetime(returns_frame["date"])
    weights_frame = weights.copy()
    weights_frame["date"] = pd.to_datetime(weights_frame["date"])

    if weights_frame.empty:
        raise ValueError("No portfolio weights were generated.")

    daily_weights = _expand_weights(
        weights_frame,
        returns_frame["date"].drop_duplicates().sort_values(),
        cost_config.holding_period_days,
    )
    daily_weights["date"] = pd.to_datetime(daily_weights["date"])
    joined = daily_weights.merge(returns_frame, on=["date", "ticker"], how="left")
    joined["gross_contribution"] = joined["weight"] * joined["daily_return"].fillna(0.0)
    joined["leg"] = np.where(joined["weight"] >= 0, "long", "short")

    # Trades are executed on the rebalance date, but the resulting book is only
    # held from the following session onward (see ``_expand_weights``). Charge
    # the cost on that first held day so it lands inside the return series.
    turnover = _compute_turnover(daily_weights)
    joined = joined.merge(turnover, on="date", how="left")
    joined["transaction_cost"] = joined["turnover"].fillna(0.0) * (cost_config.transaction_cost_bps / 10_000.0)

    daily_results = (
        joined.groupby("date", observed=True)
        .agg(
            gross_return=("gross_contribution", "sum"),
            benchmark_return=(cost_config.benchmark_column, "first"),
            turnover=("turnover", "first"),
        )
        .reset_index()
    )
    daily_results["net_return"] = daily_results["gross_return"] - daily_results["turnover"].fillna(0.0) * (
        cost_config.transaction_cost_bps / 10_000.0
    )
    daily_results["cumulative_return"] = (1.0 + daily_results["net_return"]).cumprod() - 1.0
    daily_results["drawdown"] = (
        (1.0 + daily_results["net_return"]).cumprod()
        / (1.0 + daily_results["net_return"]).cumprod().cummax()
        - 1.0
    )

    metrics = _compute_metrics(daily_results, cost_config)
    leg_contributions = (
        joined.groupby(["date", "leg"], observed=True)["gross_contribution"].sum().reset_index()
    )
    regime_metrics = _compute_regime_metrics(daily_results, cost_config)
    return BacktestResult(
        daily_results=daily_results,
        metrics=metrics,
        leg_contributions=leg_contributions,
        regime_metrics=regime_metrics,
        weights=weights_frame,
    )


def save_backtest_result(result: BacktestResult, output_dir: str | Path) -> None:
    """Persist all BacktestResult dataframes as CSVs under ``output_dir``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.daily_results.to_csv(output_path / "daily_results.csv", index=False)
    result.metrics.to_csv(output_path / "metrics.csv", index=False)
    result.leg_contributions.to_csv(output_path / "leg_contributions.csv", index=False)
    result.regime_metrics.to_csv(output_path / "regime_metrics.csv", index=False)
    result.weights.to_csv(output_path / "weights.csv", index=False)


def _expand_weights(
    weights: pd.DataFrame, available_dates: pd.Series, holding_period: int
) -> pd.DataFrame:
    """Broadcast each rebalance-date book over the sessions it is actually held.

    ``daily_return`` on date ``t`` is the realised move from ``t-1`` to ``t``, so
    a book formed from information available on rebalance date ``t0`` earns its
    first return on the *following* session. Each book therefore spans
    ``(t0, t0 + holding_period]``. Starting at ``t0`` itself would credit the new
    book with a return realised before it existed, which inverts any reversal
    component of the signal.

    When books are formed more often than ``holding_period``, several are live at
    once. That is the intended overlapping-tranche behaviour: the portfolio on any
    date is the equal-weighted average of every book still inside its holding
    window, so each rebalance turns over only a fraction of the position.
    """
    weights = weights.sort_values(["date", "ticker"]).copy()
    available_dates = list(pd.to_datetime(available_dates))
    date_lookup = {date: idx for idx, date in enumerate(available_dates)}
    holding_period = max(1, holding_period)
    rows: list[pd.DataFrame] = []

    for start_date, block in weights.groupby("date", sort=True):
        start_idx = date_lookup.get(start_date)
        if start_idx is None:
            continue
        held_dates = available_dates[start_idx + 1 : start_idx + 1 + holding_period]
        if not held_dates:
            continue
        block = block.copy()
        for held_date in held_dates:
            expanded = block.copy()
            expanded["date"] = held_date
            rows.append(expanded)

    if not rows:
        return pd.DataFrame(columns=weights.columns)

    expanded = pd.concat(rows, ignore_index=True)
    # Average across the live tranches, then re-normalise so the blended book
    # still carries unit gross exposure. Names held by several tranches keep the
    # larger position, which is the point: persistent conviction gets more capital.
    live_tranches = expanded.groupby("date")["ticker"].transform("size") / expanded.groupby(
        "date"
    )["ticker"].transform("nunique")
    expanded["weight"] = expanded["weight"] / live_tranches.clip(lower=1.0)
    expanded = expanded.groupby(["date", "ticker"], as_index=False).agg(
        weight=("weight", "sum"),
        prediction=("prediction", "mean"),
        sector=("sector", "first"),
        beta_60d=("beta_60d", "mean"),
        split=("split", "first"),
    )
    gross = expanded.groupby("date")["weight"].transform(lambda w: w.abs().sum())
    expanded["weight"] = expanded["weight"] / gross.where(gross > 0, 1.0)
    return expanded


def _compute_turnover(weights: pd.DataFrame) -> pd.DataFrame:
    """Compute one-way turnover per rebalance date as sum of absolute weight changes.

    The first rebalance is charged the full gross exposure, since establishing the
    initial book is a real trade. ``min_count=1`` is what makes that happen: a
    plain ``sum`` over the all-NaN first ``diff`` row returns 0.0 rather than NaN,
    so the ``fillna`` fallback never fires and the opening trade comes out free.
    """
    pivot = (
        weights.pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    turnover = pivot.diff().abs().sum(axis=1, min_count=1).fillna(pivot.abs().sum(axis=1))
    return turnover.rename("turnover").reset_index()


def _compute_metrics(daily_results: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Compute annualised summary metrics: return, vol, Sharpe, drawdown, hit rate, turnover."""
    net = daily_results["net_return"].fillna(0.0)
    benchmark = daily_results["benchmark_return"].fillna(0.0)
    annualized_return = (1.0 + net.mean()) ** config.annualization_factor - 1.0
    annualized_vol = net.std(ddof=0) * np.sqrt(config.annualization_factor)
    sharpe = annualized_return / annualized_vol if annualized_vol else np.nan
    hit_rate = (net > 0).mean()
    max_drawdown = daily_results["drawdown"].min()
    info_coef = np.nan
    if net.std(ddof=0) > 0 and benchmark.std(ddof=0) > 0:
        info_coef = net.corr(benchmark)
    mean_turnover = daily_results["turnover"].fillna(0.0).mean()
    implied_holding_period = np.nan if mean_turnover == 0 else 1.0 / mean_turnover

    return pd.DataFrame(
        [
            {
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "hit_rate": hit_rate,
                "information_coefficient_proxy": info_coef,
                "average_turnover": mean_turnover,
                "implied_holding_period_days": implied_holding_period,
            }
        ]
    )


def _compute_regime_metrics(daily_results: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Split performance by high- vs. low-volatility regimes using rolling benchmark vol."""
    frame = daily_results.copy()
    frame["benchmark_vol_20d"] = frame["benchmark_return"].rolling(config.regime_vol_window).std()
    median_vol = frame["benchmark_vol_20d"].median(skipna=True)
    frame["regime"] = np.where(frame["benchmark_vol_20d"] >= median_vol, "high_vol", "low_vol")
    regime_metrics = (
        frame.groupby("regime", observed=True)["net_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "average_return", "std": "return_std", "count": "days"})
    )
    return regime_metrics
