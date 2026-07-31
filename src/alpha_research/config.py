from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from . import universe


def _default_tickers() -> list[str]:
    """Full S&P Composite 1500 membership, excluding the benchmark itself."""
    return [ticker for ticker in universe.sp1500().tickers if ticker != "SPY"]


@dataclass(slots=True)
class DataConfig:
    source: str = "yfinance"
    # A long sample is not a luxury here. A 21-session strategy produces roughly
    # twelve independent bets a year, so a four-year window cannot distinguish a
    # real effect from noise no matter how the Sharpe is annualised.
    start_date: str = "2004-01-01"
    end_date: str = "2024-12-31"
    benchmark: str = "SPY"
    universe_name: str = "sp1500"
    tickers: list[str] = field(default_factory=_default_tickers)
    cache_dir: str = "data/cache"
    min_history_days: int = 252
    min_price: float = 5.0
    min_median_dollar_volume: float = 10_000_000.0
    # Bulk requests for a universe this size get throttled; batching plus retries
    # is what keeps a throttled slice from silently vanishing from the panel.
    download_batch_size: int = 100
    download_max_attempts: int = 4
    download_retry_seconds: float = 20.0
    synthetic_seed: int = 7
    synthetic_tickers: int = 60


@dataclass(slots=True)
class FeatureConfig:
    winsorize_quantiles: tuple[float, float] = (0.02, 0.98)
    rank_features: list[str] = field(
        default_factory=lambda: [
            "reversal_1d",
            "reversal_5d",
            "reversal_21d",
            "momentum_20d",
            "momentum_60d",
            "momentum_12_1",
            "log_dollar_volume",
            "vol_adj_momentum_20d",
            "abnormal_volume_20d",
            "turnover_ratio_20d",
            "beta_60d",
            "beta_instability_20d",
            "idio_vol_60d",
            "extreme_move_reversal_flag",
            "momentum_6_1",
            "high_52w_proximity",
            "price_to_ma_200d",
            "max_daily_return_21d",
            "amihud_illiquidity_21d",
            "realized_vol_60d",
            "realized_vol_252d",
            "return_skew_60d",
            "volume_variability_60d",
            "dollar_volume_trend_60d",
            "up_day_share_60d",
            "close_location_21d",
        ]
    )
    target_horizons: list[int] = field(default_factory=lambda: [5, 21])
    benchmark_window: int = 60
    volatility_window: int = 20


@dataclass(slots=True)
class SplitConfig:
    validation_start: str = "2016-01-04"
    test_start: str = "2020-01-02"
    retrain_frequency_days: int = 21
    min_train_observations: int = 252
    # Subsample training dates. Consecutive daily observations of a 21-session
    # forward return overlap in 20 of their 21 days, so adjacent rows are close
    # to duplicates: they inflate the apparent sample size without adding
    # independent information, while multiplying fit cost. Taking every k-th
    # date keeps the coverage and drops the redundancy. Set to 1 to disable.
    train_sample_every_n_sessions: int = 5


@dataclass(slots=True)
class ModelConfig:
    # Gradient boosting, selected on the validation split. It roughly doubled the
    # information coefficient against ridge on both out-of-sample windows
    # (validation 0.021 -> 0.041, test 0.013 -> 0.027), which matches the
    # published finding that the gains from machine learning on this problem come
    # from nonlinear interactions between characteristics rather than from any
    # single stronger predictor.
    model_type: str = "gbm"
    target_column: str = "target_residual_21d_z"
    # Beta, beta instability, and idiosyncratic volatility are deliberately absent.
    # With them in the design matrix the model reliably discovers that high-beta,
    # high-volatility names outperform in a rising training window and builds a
    # market bet: the long-minus-short beta of the resulting book measured +0.50,
    # and essentially all of its out-of-sample P&L came from that tilt rather than
    # from stock selection. They remain in the panel for use as neutralisation
    # constraints in portfolio construction, which is where risk exposures belong.
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "reversal_1d_rank",
            "reversal_5d_rank",
            "reversal_21d_rank",
            "momentum_20d_rank",
            "momentum_60d_rank",
            "momentum_12_1_rank",
            "log_dollar_volume_rank",
            "vol_adj_momentum_20d_rank",
            "abnormal_volume_20d_rank",
            "turnover_ratio_20d_rank",
            "extreme_move_reversal_flag_rank",
            "momentum_6_1_rank",
            "high_52w_proximity_rank",
            "price_to_ma_200d_rank",
            "max_daily_return_21d_rank",
            "amihud_illiquidity_21d_rank",
            "realized_vol_60d_rank",
            "realized_vol_252d_rank",
            "return_skew_60d_rank",
            "volume_variability_60d_rank",
            "dollar_volume_trend_60d_rank",
            "up_day_share_60d_rank",
            "close_location_21d_rank",
        ]
    )
    ridge_alpha: float = 2.0
    random_state: int = 7
    # Fraction of the prediction's linear span in the features that is projected
    # out, within each date. Trading raw correlation for stability: with no
    # neutralisation the model leans hard on whichever features worked in the
    # training window, and the validation-to-test Sharpe gap was 0.91. At 0.50
    # that gap falls to 0.63 and the untouched holdout Sharpe *rises* from 0.66
    # to 0.73, because a period in which one feature stops paying does less
    # damage. Selected on validation from {0, 0.25, 0.5, 0.75, 1}.
    feature_neutralization: float = 0.50
    # Gradient boosting hyperparameters. Shallow trees, a slow learning rate, and
    # large leaves are all deliberate: monthly cross-sectional return prediction
    # has an R-squared near zero, so capacity is far more likely to buy noise
    # than signal. These were fixed a priori rather than searched, since tuning
    # them against the evaluation window is exactly the failure mode this
    # project's validation protocol exists to prevent.
    gbm_max_depth: int = 3
    gbm_max_iter: int = 300
    gbm_learning_rate: float = 0.03
    gbm_min_samples_leaf: int = 500
    gbm_l2_regularization: float = 1.0
    gbm_max_features: float = 0.7
    # Signs follow the measured standalone rank IC of each feature against the
    # 21-session beta-residual forward return, not intuition. Note momentum_60d:
    # a 60-session window sits inside the short-term reversal region and its IC is
    # negative (t = -3.0), so it carries a negative weight despite the name.
    composite_signal_weights: dict[str, float] = field(
        default_factory=lambda: {
            "momentum_12_1_rank": 0.30,
            "log_dollar_volume_rank": -0.25,
            "momentum_60d_rank": -0.15,
            "abnormal_volume_20d_rank": 0.15,
            "turnover_ratio_20d_rank": -0.10,
            "reversal_5d_rank": 0.05,
        }
    )


@dataclass(slots=True)
class PortfolioConfig:
    # A wider book than the conventional decile. Validation preferred 20% on both
    # sides: the extra names dilute average signal strength but add breadth, and
    # breadth wins here because the per-name information coefficient is small.
    long_quantile: float = 0.20
    short_quantile: float = 0.20
    # Matched to the 21-session prediction horizon. Rebalancing faster than the
    # signal decays pays turnover for information the model has not refreshed: at
    # a 5-session cadence the book turned over 0.32 of gross per day, which at
    # 10 bps costs more than the strategy's entire gross alpha.
    rebalance_frequency_days: int = 21
    sector_neutral: bool = True
    beta_neutral: bool = True
    min_names_per_side: int = 5
    # Number of overlapping books. Each is rebalanced on a staggered schedule and
    # held for the full rebalance_frequency_days, so only 1/tranches of the
    # position turns over at a time. Chosen on validation from {1, 7, 21}.
    tranches: int = 7
    beta_column: str = "beta_60d"
    sector_column: str = "sector"


@dataclass(slots=True)
class BacktestConfig:
    transaction_cost_bps: float = 10.0
    # Sessions each book is held. Must match PortfolioConfig.rebalance_frequency_days;
    # the backtest needs it explicitly because staggered tranches mean the gap
    # between consecutive rebalance dates is no longer the holding period.
    holding_period_days: int = 21
    annualization_factor: int = 252
    benchmark_column: str = "benchmark_return"
    regime_vol_window: int = 20


@dataclass(slots=True)
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ResearchConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    config = ResearchConfig()
    for section_name, dataclass_type in (
        ("data", config.data),
        ("features", config.features),
        ("split", config.split),
        ("model", config.model),
        ("portfolio", config.portfolio),
        ("backtest", config.backtest),
    ):
        if section_name in raw:
            _merge_dataclass(dataclass_type, raw[section_name])
    return config
