"""End-to-end smoke test: full pipeline from data build through backtest metrics."""

from __future__ import annotations

from alpha_research.backtest import run_backtest
from alpha_research.config import ResearchConfig
from alpha_research.data import build_dataset
from alpha_research.features import generate_features
from alpha_research.modeling import fit_predict
from alpha_research.portfolio import construct_portfolio


def _make_config() -> ResearchConfig:
    config = ResearchConfig()
    config.data.source = "synthetic"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2024-12-31"
    config.data.synthetic_tickers = 60
    config.split.validation_start = "2022-01-03"
    config.split.test_start = "2023-01-03"
    # 10% of 60 = 6 names per side, above the default min of 5
    config.portfolio.long_quantile = 0.10
    config.portfolio.short_quantile = 0.10
    return config


def test_full_pipeline_produces_nonempty_metrics() -> None:
    """The entire pipeline should run without error and return a non-empty metrics table."""
    config = _make_config()

    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    predictions = fit_predict(features, config.split, config.model)
    exposures = features[["date", "ticker", "sector", "beta_60d"]].drop_duplicates(["date", "ticker"])
    weights = construct_portfolio(predictions, exposures, config.portfolio)
    result = run_backtest(weights, features, config.backtest)

    assert not result.metrics.empty
    assert "sharpe_ratio" in result.metrics.columns
    assert not result.daily_results.empty
    assert not result.leg_contributions.empty


def test_full_pipeline_predictions_cover_test_split() -> None:
    config = _make_config()
    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    predictions = fit_predict(features, config.split, config.model)

    assert "test" in predictions["split"].values


def test_full_pipeline_weights_have_both_legs() -> None:
    """Every rebalance date must have at least one long and one short position."""
    config = _make_config()
    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    predictions = fit_predict(features, config.split, config.model)
    exposures = features[["date", "ticker", "sector", "beta_60d"]].drop_duplicates(["date", "ticker"])
    weights = construct_portfolio(predictions, exposures, config.portfolio)

    for date, group in weights.groupby("date"):
        assert (group["weight"] > 0).any(), f"No long positions on {date}"
        assert (group["weight"] < 0).any(), f"No short positions on {date}"


def test_full_pipeline_regime_metrics_has_two_regimes() -> None:
    config = _make_config()
    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    predictions = fit_predict(features, config.split, config.model)
    exposures = features[["date", "ticker", "sector", "beta_60d"]].drop_duplicates(["date", "ticker"])
    weights = construct_portfolio(predictions, exposures, config.portfolio)
    result = run_backtest(weights, features, config.backtest)

    assert set(result.regime_metrics["regime"]) == {"high_vol", "low_vol"}
