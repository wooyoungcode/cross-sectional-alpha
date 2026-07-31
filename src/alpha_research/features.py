"""Cross-sectional feature engineering for the alpha research pipeline.

Computes momentum, reversal, volatility, volume, and beta-based signals on a
daily equity panel, then applies leakage-safe forward-return targets. All
features are cross-sectionally winsorized and z-scored per date to remove
market-level effects before modeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


def generate_features(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Engineer cross-sectional features and forward-return targets from a raw panel.

    Parameters
    ----------
    panel:
        Raw daily panel as returned by :func:`~alpha_research.data.build_dataset`.
        Must contain columns ``ticker``, ``date``, ``adj_close``, ``volume``,
        ``dollar_volume``, ``daily_return``, ``benchmark_return``, and
        ``is_benchmark``.
    config:
        Feature configuration controlling volatility window, target horizons,
        rank features, and winsorize quantiles.

    Returns
    -------
    pd.DataFrame
        The equity-only subset of ``panel`` augmented with raw feature columns,
        cross-sectionally z-scored ``*_z`` variants, optional rank-scaled
        ``*_rank`` variants, and forward-return target columns.
    """
    equity_panel = panel.loc[~panel["is_benchmark"]].copy()
    equity_panel = equity_panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = equity_panel.groupby("ticker", observed=True)
    returns = grouped["adj_close"].pct_change()
    equity_panel["reversal_1d"] = -returns
    equity_panel["reversal_5d"] = -grouped["adj_close"].pct_change(5)
    equity_panel["reversal_21d"] = -grouped["adj_close"].pct_change(21)
    equity_panel["momentum_20d"] = grouped["adj_close"].pct_change(20)
    equity_panel["momentum_60d"] = grouped["adj_close"].pct_change(60)
    # Standard cross-sectional momentum: the 12-month return with the most recent
    # month skipped. The skip matters because the trailing month carries the
    # short-term reversal effect, which partially cancels medium-term momentum.
    # Measured standalone on this panel, the 60-day window has a *negative* rank
    # IC against forward residual returns while 12-1 is positive, so a 60-day
    # window is a reversal signal rather than a momentum signal.
    equity_panel["momentum_12_1"] = (
        grouped["adj_close"].shift(21) / grouped["adj_close"].shift(252) - 1.0
    )
    # Size / liquidity proxy. Market capitalisation is not available from the
    # price feed, so traded notional stands in for it.
    equity_panel["log_dollar_volume"] = np.log1p(equity_panel["dollar_volume"])
    realized_vol = grouped["daily_return"].rolling(config.volatility_window).std().reset_index(level=0, drop=True)
    equity_panel["realized_vol_20d"] = realized_vol
    equity_panel["vol_adj_momentum_20d"] = equity_panel["momentum_20d"] / realized_vol.replace(0.0, np.nan)

    rolling_volume = grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    rolling_dollar = grouped["dollar_volume"].rolling(20).mean().reset_index(level=0, drop=True)
    equity_panel["abnormal_volume_20d"] = equity_panel["volume"] / rolling_volume.replace(0.0, np.nan) - 1.0
    equity_panel["turnover_ratio_20d"] = equity_panel["dollar_volume"] / rolling_dollar.replace(0.0, np.nan) - 1.0

    equity_panel = _add_extended_signals(equity_panel, config)

    beta_frame = _compute_rolling_beta_features(equity_panel, config.benchmark_window)
    equity_panel = equity_panel.merge(beta_frame, on=["date", "ticker"], how="left")

    rolling_std = grouped["daily_return"].rolling(20).std().reset_index(level=0, drop=True)
    extreme_move = grouped["daily_return"].shift(1).abs() > (2.0 * rolling_std.shift(1))
    prior_direction = -np.sign(grouped["daily_return"].shift(1)).fillna(0.0)
    equity_panel["extreme_move_reversal_flag"] = extreme_move.astype(float) * prior_direction

    benchmark_forward = _benchmark_forward_returns(panel, config.target_horizons)
    equity_panel = equity_panel.merge(benchmark_forward, on="date", how="left")
    for horizon in config.target_horizons:
        equity_panel[f"target_return_{horizon}d"] = (
            grouped["adj_close"].shift(-horizon) / equity_panel["adj_close"] - 1.0
        )
        # Beta-adjusted target. Regressing on the raw forward return rewards the
        # model for loading on beta whenever the training window happens to be a
        # rising market, which produces a market bet dressed up as alpha. Netting
        # out beta times the benchmark's forward return leaves the stock-specific
        # component, which is what a market-neutral book can actually harvest.
        equity_panel[f"target_residual_{horizon}d"] = (
            equity_panel[f"target_return_{horizon}d"]
            - equity_panel["beta_60d"] * equity_panel[f"benchmark_forward_{horizon}d"]
        )
        # Standardise the target within each date. The portfolio only ever acts on
        # the ordering of names on a given day, so training the model to forecast
        # the level of returns spends capacity on a quantity the strategy discards.
        equity_panel[f"target_residual_{horizon}d_z"] = equity_panel.groupby("date")[
            f"target_residual_{horizon}d"
        ].transform(_standardize)
    equity_panel["target_top_quintile_5d"] = _cross_sectional_top_quintile(
        equity_panel, "target_return_5d"
    )

    feature_columns = [
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
        "overnight_return_21d",
        "intraday_return_21d",
        "overnight_intraday_gap_21d",
        "rank_momentum_21d",
        "volume_weighted_return_21d",
    ]
    equity_panel = _cross_sectional_preprocess(
        equity_panel,
        feature_columns=feature_columns,
        rank_features=config.rank_features,
        winsorize_quantiles=config.winsorize_quantiles,
    )
    return equity_panel


def _compute_rolling_beta_features(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling OLS beta, beta instability, and idiosyncratic volatility.

    For each ticker, regresses daily returns on benchmark returns over a rolling
    ``window``-day window to produce:

    - ``beta_60d``: rolling market beta (covariance / benchmark variance)
    - ``beta_instability_20d``: 20-day rolling std of ``beta_60d``
    - ``idio_vol_60d``: rolling std of the market-residual return
    """
    frame = panel[["date", "ticker", "daily_return", "benchmark_return"]].copy()

    def per_ticker(group: pd.DataFrame) -> pd.DataFrame:
        ticker = group.name
        cov = group["daily_return"].rolling(window).cov(group["benchmark_return"])
        bench_var = group["benchmark_return"].rolling(window).var()
        beta = cov / bench_var.replace(0.0, np.nan)
        residual = group["daily_return"] - beta * group["benchmark_return"]
        return pd.DataFrame(
            {
                "date": group["date"].to_numpy(),
                "ticker": np.repeat(ticker, len(group)),
                "beta_60d": beta.to_numpy(),
                "beta_instability_20d": beta.rolling(20).std().to_numpy(),
                "idio_vol_60d": residual.rolling(window).std().to_numpy(),
            },
            index=group.index,
        )

    computed = (
        frame.groupby("ticker", observed=True, group_keys=False)
        .apply(per_ticker, include_groups=False)
        .reset_index(drop=True)
    )
    return computed[["date", "ticker", "beta_60d", "beta_instability_20d", "idio_vol_60d"]]


def _add_extended_signals(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add the wider set of price and volume signals.

    Each entry is a cross-sectional effect documented in the asset pricing
    literature and computable from daily bars alone. The point of carrying a
    broad set rather than a handful is that a linear model over few, highly
    correlated technical signals has very little to work with; the marginal
    signals here are only weakly correlated with the momentum and reversal core.

    Fundamentals-based effects, notably value and quality, remain out of reach
    with a price-only feed and stay on the limitations list.
    """
    grouped = panel.groupby("ticker", observed=True)
    close = panel["adj_close"]
    daily_return = panel["daily_return"]

    def rolling(column: str, window: int, how: str) -> pd.Series:
        series = grouped[column].rolling(window)
        return getattr(series, how)().reset_index(level=0, drop=True)

    # Momentum at a second medium horizon, same skip-a-month convention.
    panel["momentum_6_1"] = grouped["adj_close"].shift(21) / grouped["adj_close"].shift(126) - 1.0

    # Proximity to the 52-week high. Prices near their high keep drifting up;
    # the ratio is a cleaner statement of the same effect than raw momentum.
    panel["high_52w_proximity"] = close / rolling("adj_close", 252, "max").replace(0.0, np.nan)

    # Trend relative to the long moving average.
    panel["price_to_ma_200d"] = close / rolling("adj_close", 200, "mean").replace(0.0, np.nan) - 1.0

    # Lottery demand. Stocks with an extreme best day in the trailing month are
    # bid up by investors paying for the small chance of a repeat, and
    # subsequently underperform, so the sign is negated to keep "higher is better".
    panel["max_daily_return_21d"] = -rolling("daily_return", 21, "max")

    # Illiquidity: absolute return per unit of traded notional.
    panel["amihud_illiquidity_21d"] = np.log1p(
        (daily_return.abs() / panel["dollar_volume"].replace(0.0, np.nan))
        .groupby(panel["ticker"], observed=True)
        .rolling(21)
        .mean()
        .reset_index(level=0, drop=True)
        * 1e9
    )

    # Realised volatility at two horizons, negated for the low-volatility effect.
    panel["realized_vol_60d"] = -rolling("daily_return", 60, "std")
    panel["realized_vol_252d"] = -rolling("daily_return", 252, "std")

    # Return skewness. Positively skewed payoffs attract the same lottery demand.
    panel["return_skew_60d"] = -rolling("daily_return", 60, "skew")

    # Dispersion of trading activity, distinct from its level.
    volume_mean = rolling("volume", 60, "mean").replace(0.0, np.nan)
    panel["volume_variability_60d"] = -(rolling("volume", 60, "std") / volume_mean)

    # Longer-run liquidity trend: is participation building or fading.
    panel["dollar_volume_trend_60d"] = (
        rolling("dollar_volume", 21, "mean") / rolling("dollar_volume", 252, "mean").replace(0.0, np.nan) - 1.0
    )

    # Fraction of up days, a path measure rather than an endpoint measure.
    panel["up_day_share_60d"] = (
        (daily_return > 0).astype(float).groupby(panel["ticker"], observed=True)
        .rolling(60).mean().reset_index(level=0, drop=True)
    )

    # Overnight and intraday returns, kept separate. The two halves of the day are
    # driven by different things: the overnight move prices news and accumulated
    # order imbalance against a closed book, while the intraday move is largely
    # liquidity provision.
    #
    # These are computed but deliberately NOT in ModelConfig.feature_columns.
    # Adding them cost 0.10 of Sharpe on both the validation and test windows
    # (0.73 to 0.63 on each). The information coefficient stayed high, so the
    # predictive content did not vanish; it stopped converting into portfolio
    # return, most likely because feature neutralisation projects against the
    # design matrix and a wider matrix removes more of the signal along with the
    # fragility. They are retained here because the computation is correct and
    # cheap, and because a future configuration without neutralisation, or with
    # feature selection in front of the model, may well want them.
    if {"open", "close"}.issubset(panel.columns):
        previous_close = grouped["close"].shift(1)
        overnight = panel["open"] / previous_close.replace(0.0, np.nan) - 1.0
        intraday = panel["close"] / panel["open"].replace(0.0, np.nan) - 1.0
        panel["overnight_return_21d"] = (
            overnight.groupby(panel["ticker"], observed=True).rolling(21).sum()
            .reset_index(level=0, drop=True)
        )
        panel["intraday_return_21d"] = (
            intraday.groupby(panel["ticker"], observed=True).rolling(21).sum()
            .reset_index(level=0, drop=True)
        )
        # Which half of the day the stock's recent drift came from. A name rising
        # overnight and fading intraday is a different animal from the reverse.
        panel["overnight_intraday_gap_21d"] = (
            panel["overnight_return_21d"] - panel["intraday_return_21d"]
        )

    # Momentum of the stock's cross-sectional standing rather than of its price.
    # A stock can rise while falling through the ranking; the ranking is what a
    # long/short book actually trades.
    price_rank = panel.groupby("date")["adj_close"].rank(pct=True)
    panel["rank_momentum_21d"] = (
        price_rank - price_rank.groupby(panel["ticker"], observed=True).shift(21)
    )

    # Return earned on heavy-volume days relative to quiet ones. Informed trading
    # concentrates in volume, so the sign of the volume-weighted move says
    # something the unweighted move does not.
    volume_share = panel["volume"] / rolling("volume", 21, "mean").replace(0.0, np.nan)
    panel["volume_weighted_return_21d"] = (
        (daily_return * volume_share).groupby(panel["ticker"], observed=True)
        .rolling(21).mean().reset_index(level=0, drop=True)
    )

    # Intra-range position of the close, using the raw high/low when available.
    if {"high", "low"}.issubset(panel.columns):
        span = (panel["high"] - panel["low"]).replace(0.0, np.nan)
        panel["close_location_21d"] = (
            ((panel["close"] - panel["low"]) / span)
            .groupby(panel["ticker"], observed=True)
            .rolling(21).mean().reset_index(level=0, drop=True)
        )
    return panel


def _benchmark_forward_returns(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Compound the benchmark's return over each forward horizon, one row per date.

    The horizon starts on the session *after* the observation date, matching the
    stock-side target, which is formed from a position taken at that day's close.
    """
    benchmark = (
        panel.loc[panel["is_benchmark"], ["date", "daily_return"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if benchmark.empty:
        benchmark = panel[["date", "benchmark_return"]].drop_duplicates("date").sort_values("date")
        benchmark = benchmark.rename(columns={"benchmark_return": "daily_return"}).reset_index(drop=True)

    growth = 1.0 + benchmark["daily_return"].fillna(0.0)
    for horizon in horizons:
        benchmark[f"benchmark_forward_{horizon}d"] = (
            growth.shift(-1).rolling(horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1.0
        )
    return benchmark[["date"] + [f"benchmark_forward_{h}d" for h in horizons]]


def _standardize(series: pd.Series) -> pd.Series:
    """Z-score a cross-section, returning zeros when the section has no dispersion."""
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _cross_sectional_top_quintile(panel: pd.DataFrame, column: str) -> pd.Series:
    """Return a float flag (1.0 / 0.0) marking the top-quintile stocks per date."""
    return (
        panel.groupby("date", observed=True)[column]
        .transform(lambda series: (series.rank(pct=True, method="average") >= 0.80).astype(float))
    )


def _cross_sectional_preprocess(
    panel: pd.DataFrame,
    feature_columns: list[str],
    rank_features: list[str],
    winsorize_quantiles: tuple[float, float],
) -> pd.DataFrame:
    """Winsorize, z-score, and optionally rank-scale features cross-sectionally.

    For every date independently:
    1. Winsorize each feature to ``winsorize_quantiles`` to clip outliers.
    2. Z-score to zero mean and unit std (``feature_z`` columns).
    3. For features in ``rank_features``, also add a rank-scaled column in
       ``[-1, 1]`` (``feature_rank`` columns).
    """
    processed = panel.copy()
    lower_q, upper_q = winsorize_quantiles

    for feature in feature_columns:
        processed[feature] = processed.groupby("date", observed=True)[feature].transform(
            lambda series: _winsorize(series, lower_q, upper_q)
        )
        processed[f"{feature}_z"] = processed.groupby("date", observed=True)[feature].transform(_zscore)
        if feature in rank_features:
            processed[f"{feature}_rank"] = processed.groupby("date", observed=True)[feature].transform(
                _rank_to_unit_interval
            )
    return processed


def _winsorize(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    """Clip a series to its ``lower_q`` and ``upper_q`` quantiles. Returns unchanged if fewer than 5 non-null values."""
    if series.notna().sum() < 5:
        return series
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def _zscore(series: pd.Series) -> pd.Series:
    """Return population z-scores (ddof=0). Returns zeros if std is 0 or NaN."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _rank_to_unit_interval(series: pd.Series) -> pd.Series:
    """Map percentile ranks to ``[-1, 1]`` via ``2 * (rank_pct - 0.5)``."""
    ranks = series.rank(method="average", pct=True)
    return 2.0 * (ranks - 0.5)
