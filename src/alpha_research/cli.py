"""Command-line interface for the alpha research pipeline.

Usage::

    alpha-research run --config configs/synthetic.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import save_backtest_result, run_backtest
from .config import load_config
from .data import build_dataset
from .features import generate_features
from .modeling import fit_predict
from .portfolio import construct_portfolio
from .reporting import generate_report


def run_pipeline(config_path: str) -> None:
    """Execute the full research pipeline from data ingestion to report generation."""
    config = load_config(config_path)
    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    predictions = fit_predict(features, config.split, config.model)
    exposures = features[["date", "ticker", "sector", "beta_60d"]].drop_duplicates(["date", "ticker"])
    weights = construct_portfolio(predictions, exposures, config.portfolio)
    result = run_backtest(weights, features, config.backtest)
    save_backtest_result(result, "outputs")
    generate_report(
        result,
        predictions,
        output_dir="outputs",
        report_path="docs/research_summary.md",
        features=features,
        config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cross-sectional alpha research pipeline.")
    parser.add_argument("command", choices=["run"], help="Pipeline command to execute.")
    parser.add_argument("--config", default="configs/default.toml", help="Path to a TOML config file.")
    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args.config)


if __name__ == "__main__":
    main()
