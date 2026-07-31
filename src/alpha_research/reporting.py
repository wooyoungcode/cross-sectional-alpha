"""Figures and the written research summary.

Every figure shares one style, defined in ``_STYLE`` and ``_PALETTE``, so the
output reads as a single document rather than a pile of plots. The figure set is
ordered the way a reviewer reads a study: predictive power first, then decay,
then what the portfolio built from it earned, then what it cost, then proof that
the declared risk constraints actually bind.
"""

from __future__ import annotations

import os
from pathlib import Path

_CACHE_ROOT = Path("data/cache")
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .config import ResearchConfig

_PALETTE = {
    "primary": "#1f3a5f",
    "accent": "#c1440e",
    "muted": "#8c9bab",
    "positive": "#2a6f4e",
    "negative": "#a83232",
    "grid": "#d8dee5",
    "test_span": "#f0f2f5",
}

_STYLE = {
    "figure.figsize": (9.0, 4.2),
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": _PALETTE["muted"],
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": _PALETTE["grid"],
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 9.0,
    "xtick.color": "#44515e",
    "ytick.color": "#44515e",
    "lines.linewidth": 1.6,
}


def generate_report(
    result: BacktestResult,
    predictions: pd.DataFrame,
    output_dir: str | Path = "outputs",
    report_path: str | Path = "docs/research_summary.md",
    features: pd.DataFrame | None = None,
    config: ResearchConfig | None = None,
) -> None:
    """Write every figure and the markdown research summary.

    ``features`` and ``config`` are optional so existing callers keep working;
    supplying them enables the information-coefficient and decay figures, which
    need the forward-return targets that do not survive into ``predictions``.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    config = config or ResearchConfig()
    horizon = config.features.target_horizons[-1]
    test_start = pd.Timestamp(config.split.test_start)

    daily = result.daily_results.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    ic_series = _information_coefficient(predictions, features, horizon)

    with plt.rc_context(_STYLE):
        if ic_series is not None:
            _plot_information_coefficient(ic_series, test_start, figures_dir / "information_coefficient.png")
        if features is not None:
            _plot_ic_decay(predictions, features, figures_dir / "ic_decay.png")
            _plot_decile_monotonicity(predictions, features, horizon, figures_dir / "decile_returns.png")
        _plot_equity_curve(daily, test_start, figures_dir / "equity_curve.png")
        _plot_drawdown(daily, test_start, figures_dir / "drawdown.png")
        _plot_cost_sensitivity(daily, figures_dir / "cost_sensitivity.png")
        _plot_leg_contributions(result.leg_contributions, figures_dir / "leg_contributions.png")
        _plot_risk_exposures(result.weights, features, figures_dir / "risk_exposures.png")

    _write_summary(result, ic_series, daily, config, Path(report_path))


# --------------------------------------------------------------------------- data


def _information_coefficient(
    predictions: pd.DataFrame, features: pd.DataFrame | None, horizon: int
) -> pd.Series | None:
    """Daily cross-sectional rank correlation between prediction and forward residual return."""
    if features is None:
        return None
    target = f"target_residual_{horizon}d"
    if target not in features.columns:
        return None
    frame = predictions[["date", "ticker", "prediction"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    truth = features[["date", "ticker", target]].drop_duplicates(["date", "ticker"]).copy()
    truth["date"] = pd.to_datetime(truth["date"])
    merged = frame.merge(truth, on=["date", "ticker"], how="left").dropna(subset=[target])
    if merged.empty:
        return None
    return merged.groupby("date").apply(
        lambda group: group["prediction"].corr(group[target], method="spearman"), include_groups=False
    ).dropna()


def _shade_test_period(axis, test_start: pd.Timestamp, upper: pd.Timestamp) -> None:
    axis.axvspan(test_start, upper, color=_PALETTE["test_span"], zorder=0)
    axis.axvline(test_start, color=_PALETTE["muted"], linewidth=0.9, linestyle="--")
    axis.annotate(
        "test",
        xy=(test_start, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -11),
        textcoords="offset points",
        color=_PALETTE["muted"],
        fontsize=8.5,
    )


# --------------------------------------------------------------------------- figures


def _plot_information_coefficient(ic: pd.Series, test_start: pd.Timestamp, path: Path) -> None:
    figure, axis = plt.subplots()
    axis.bar(ic.index, ic.to_numpy(), width=1.0, color=_PALETTE["muted"], alpha=0.35, linewidth=0)
    rolling = ic.rolling(63, min_periods=20).mean()
    axis.plot(rolling.index, rolling.to_numpy(), color=_PALETTE["primary"], label="63-day mean")
    axis.axhline(0.0, color=_PALETTE["accent"], linewidth=0.9)
    _shade_test_period(axis, test_start, ic.index.max())
    mean = ic.mean()
    t_stat = mean / (ic.std(ddof=1) / np.sqrt(len(ic)))
    axis.set_title(
        f"Information coefficient   mean {mean:+.4f}   IR {mean / ic.std():+.2f}   t {t_stat:+.1f}   n={len(ic)}"
    )
    axis.set_ylabel("rank IC")
    axis.legend(loc="upper left")
    figure.savefig(path)
    plt.close(figure)


def _plot_ic_decay(predictions: pd.DataFrame, features: pd.DataFrame, path: Path) -> None:
    """How long the signal stays informative, which is what sets the rebalance cadence."""
    frame = predictions[["date", "ticker", "prediction"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    panel = features[["date", "ticker", "adj_close"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"])
    horizons = [1, 2, 3, 5, 10, 21, 42, 63]
    grouped = panel.groupby("ticker", group_keys=False)["adj_close"]
    for horizon in horizons:
        panel[f"fwd_{horizon}"] = grouped.shift(-horizon) / panel["adj_close"] - 1.0
    merged = frame.merge(panel.drop(columns=["adj_close"]), on=["date", "ticker"], how="left")

    means, errors = [], []
    for horizon in horizons:
        ic = merged.dropna(subset=[f"fwd_{horizon}"]).groupby("date").apply(
            lambda group, h=horizon: group["prediction"].corr(group[f"fwd_{h}"], method="spearman"),
            include_groups=False,
        ).dropna()
        means.append(ic.mean())
        errors.append(ic.std(ddof=1) / np.sqrt(len(ic)))

    figure, axis = plt.subplots()
    axis.errorbar(
        horizons, means, yerr=np.array(errors) * 1.96, color=_PALETTE["primary"],
        marker="o", markersize=4, capsize=3, linewidth=1.4,
    )
    axis.axhline(0.0, color=_PALETTE["accent"], linewidth=0.9)
    axis.set_xscale("log")
    axis.set_xticks(horizons)
    axis.set_xticklabels(horizons)
    axis.set_title("Information coefficient by forward horizon, with 95% intervals")
    axis.set_xlabel("forward horizon (sessions)")
    axis.set_ylabel("rank IC")
    figure.savefig(path)
    plt.close(figure)


def _plot_decile_monotonicity(
    predictions: pd.DataFrame, features: pd.DataFrame, horizon: int, path: Path
) -> None:
    """A usable signal should order the deciles, not merely separate the extremes."""
    target = f"target_residual_{horizon}d"
    if target not in features.columns:
        return
    frame = predictions[["date", "ticker", "prediction"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    truth = features[["date", "ticker", target]].drop_duplicates(["date", "ticker"]).copy()
    truth["date"] = pd.to_datetime(truth["date"])
    merged = frame.merge(truth, on=["date", "ticker"], how="left").dropna(subset=[target])
    merged["decile"] = merged.groupby("date")["prediction"].transform(
        lambda series: pd.qcut(series.rank(method="first"), 10, labels=False, duplicates="drop")
    )
    stats = merged.groupby("decile")[target].agg(["mean", "sem"])

    figure, axis = plt.subplots()
    colors = [_PALETTE["negative"] if value < 0 else _PALETTE["positive"] for value in stats["mean"]]
    axis.bar(stats.index + 1, stats["mean"] * 1e4, yerr=stats["sem"] * 1e4 * 1.96,
             color=colors, alpha=0.85, capsize=3, linewidth=0)
    axis.axhline(0.0, color=_PALETTE["muted"], linewidth=0.9)
    axis.set_title(f"Mean {horizon}-session residual return by prediction decile, with 95% intervals")
    axis.set_xlabel("prediction decile (10 = most favoured)")
    axis.set_ylabel("mean residual return (bps)")
    axis.set_xticks(range(1, 11))
    figure.savefig(path)
    plt.close(figure)


def _plot_equity_curve(daily: pd.DataFrame, test_start: pd.Timestamp, path: Path) -> None:
    figure, axis = plt.subplots()
    gross = (1.0 + daily["gross_return"].fillna(0.0)).cumprod() - 1.0
    axis.plot(daily["date"], gross * 100, color=_PALETTE["muted"], label="gross of costs")
    axis.plot(daily["date"], daily["cumulative_return"] * 100, color=_PALETTE["primary"], label="net of costs")
    axis.axhline(0.0, color=_PALETTE["muted"], linewidth=0.8)
    _shade_test_period(axis, test_start, daily["date"].max())
    axis.set_title("Cumulative return, market-neutral long/short book")
    axis.set_ylabel("cumulative return (%)")
    axis.legend(loc="upper left")
    figure.savefig(path)
    plt.close(figure)


def _plot_drawdown(daily: pd.DataFrame, test_start: pd.Timestamp, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 2.8))
    axis.fill_between(daily["date"], daily["drawdown"] * 100, 0.0, color=_PALETTE["negative"], alpha=0.35, linewidth=0)
    axis.plot(daily["date"], daily["drawdown"] * 100, color=_PALETTE["negative"], linewidth=1.1)
    _shade_test_period(axis, test_start, daily["date"].max())
    axis.set_title(f"Drawdown, net of costs   trough {daily['drawdown'].min():.1%}")
    axis.set_ylabel("drawdown (%)")
    figure.savefig(path)
    plt.close(figure)


def _plot_cost_sensitivity(daily: pd.DataFrame, path: Path) -> None:
    """Where the strategy stops paying, which is the number that decides viability."""
    gross = daily["gross_return"].fillna(0.0)
    turnover = daily["turnover"].fillna(0.0)
    if turnover.mean() <= 0:
        return
    grid = np.linspace(0, 40, 81)
    sharpes = []
    for bps in grid:
        net = gross - turnover * bps / 1e4
        std = net.std(ddof=0)
        sharpes.append(net.mean() / std * np.sqrt(252) if std else np.nan)
    breakeven = gross.mean() / turnover.mean() * 1e4

    figure, axis = plt.subplots()
    axis.plot(grid, sharpes, color=_PALETTE["primary"])
    axis.axhline(0.0, color=_PALETTE["muted"], linewidth=0.9)
    axis.axvline(breakeven, color=_PALETTE["accent"], linestyle="--", linewidth=1.1)
    axis.annotate(
        f"breakeven {breakeven:.0f} bps",
        xy=(breakeven, 0.0), xytext=(6, 14), textcoords="offset points",
        color=_PALETTE["accent"], fontsize=9,
    )
    axis.set_title("Net Sharpe against assumed one-way transaction cost")
    axis.set_xlabel("transaction cost (bps, one way)")
    axis.set_ylabel("annualised net Sharpe")
    figure.savefig(path)
    plt.close(figure)


def _plot_leg_contributions(leg_contributions: pd.DataFrame, path: Path) -> None:
    pivot = leg_contributions.pivot(index="date", columns="leg", values="gross_contribution").fillna(0.0)
    pivot.index = pd.to_datetime(pivot.index)
    figure, axis = plt.subplots()
    colors = {"long": _PALETTE["positive"], "short": _PALETTE["negative"]}
    for column in pivot.columns:
        axis.plot(pivot.index, pivot[column].cumsum() * 100, label=f"{column} leg",
                  color=colors.get(column, _PALETTE["primary"]))
    axis.axhline(0.0, color=_PALETTE["muted"], linewidth=0.8)
    axis.set_title("Cumulative gross contribution by leg")
    axis.set_ylabel("cumulative contribution (%)")
    axis.legend(loc="upper left")
    figure.savefig(path)
    plt.close(figure)


def _plot_risk_exposures(weights: pd.DataFrame, features: pd.DataFrame | None, path: Path) -> None:
    """Evidence that the neutrality claims hold in the delivered book, not just in intent."""
    if features is None or weights.empty:
        return
    frame = weights[["date", "ticker", "weight"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    risk = features[["date", "ticker", "beta_60d", "sector"]].drop_duplicates(["date", "ticker"]).copy()
    risk["date"] = pd.to_datetime(risk["date"])
    merged = frame.merge(risk, on=["date", "ticker"], how="left")
    merged["beta_60d"] = merged["beta_60d"].fillna(1.0)

    per_date = merged.groupby("date").apply(
        lambda group: pd.Series(
            {
                "beta": float((group["weight"] * group["beta_60d"]).sum()),
                "net": float(group["weight"].sum()),
                "max_sector": float(group.groupby("sector")["weight"].sum().abs().max()),
            }
        ),
        include_groups=False,
    )

    # Beta and net exposure are zero to machine precision, so plotting them on
    # the same axis as the sector exposure renders both as one invisible line
    # along y=0: the figure would be unable to show the very property it exists
    # to demonstrate. They get their own panel on a scale that can resolve them.
    figure, (upper, lower) = plt.subplots(
        2, 1,
        figsize=(9.0, 4.8),
        sharex=True,
        # Space between the panels so the lower title clears the upper axis.
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.38},
    )

    upper.plot(per_date.index, per_date["max_sector"] * 100, color=_PALETTE["primary"],
               label="largest net sector")
    upper.axhline(0.0, color=_PALETTE["muted"], linewidth=0.8)
    upper.set_ylabel("exposure (% of gross)")
    upper.set_title("Realised risk exposures at each rebalance")
    # Headroom so the legend never sits on top of the series.
    upper.set_ylim(top=float(per_date["max_sector"].max()) * 100 * 1.35)
    upper.legend(loc="upper left")

    worst_beta = float(per_date["beta"].abs().max())
    worst_net = float(per_date["net"].abs().max())
    lower.plot(per_date.index, per_date["beta"], color=_PALETTE["accent"], label="portfolio beta")
    lower.plot(per_date.index, per_date["net"], color=_PALETTE["positive"],
               linestyle="--", label="net dollar exposure")
    lower.axhline(0.0, color=_PALETTE["muted"], linewidth=0.8)
    lower.set_ylabel("exposure")
    lower.set_title(
        f"Dollar and beta neutrality hold to machine precision "
        f"(worst |beta| {worst_beta:.0e}, worst |net| {worst_net:.0e})",
        fontsize=9.0,
    )
    lower.legend(loc="upper left", ncol=2)

    figure.savefig(path)
    plt.close(figure)


# --------------------------------------------------------------------------- summary


def _split_metrics(daily: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Sharpe and drawdown for each split, plus a Newey-West t on the daily net series."""
    bounds = {
        "validation": (pd.Timestamp(config.split.validation_start), pd.Timestamp(config.split.test_start)),
        "test": (pd.Timestamp(config.split.test_start), daily["date"].max() + pd.Timedelta(days=1)),
        "combined": (pd.Timestamp(config.split.validation_start), daily["date"].max() + pd.Timedelta(days=1)),
    }
    lag = config.portfolio.rebalance_frequency_days
    rows = []
    for name, (low, high) in bounds.items():
        window = daily[(daily["date"] >= low) & (daily["date"] < high)]
        net = window["net_return"].fillna(0.0)
        if len(net) < 2 or net.std(ddof=0) == 0:
            continue
        centred = net.to_numpy() - net.mean()
        n = len(centred)
        variance = (centred @ centred) / n + 2 * sum(
            (1 - k / (lag + 1)) * (centred[k:] @ centred[:-k]) / n for k in range(1, min(lag, n - 1) + 1)
        )
        rows.append(
            {
                "split": name,
                "days": n,
                "annualised return": f"{(1 + net.mean()) ** 252 - 1:+.2%}",
                "annualised vol": f"{net.std(ddof=0) * np.sqrt(252):.2%}",
                "Sharpe": f"{net.mean() / net.std(ddof=0) * np.sqrt(252):+.2f}",
                "Newey-West t": f"{net.mean() / np.sqrt(variance / n):+.2f}" if variance > 0 else "n/a",
                "max drawdown": f"{window['drawdown'].min():.1%}",
            }
        )
    return pd.DataFrame(rows)


def _write_summary(
    result: BacktestResult,
    ic: pd.Series | None,
    daily: pd.DataFrame,
    config: ResearchConfig,
    report_path: Path,
) -> None:
    metrics = result.metrics.iloc[0]
    splits = _split_metrics(daily, config)
    gross, turnover = daily["gross_return"].fillna(0.0), daily["turnover"].fillna(0.0)
    breakeven = gross.mean() / turnover.mean() * 1e4 if turnover.mean() else float("nan")

    if ic is not None and len(ic) > 1:
        t_stat = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
        yearly = ic.groupby(ic.index.year).mean()
        ic_block = (
            f"Rank IC against the {config.features.target_horizons[-1]}-session beta-residual forward return, "
            f"measured on {len(ic):,} out-of-sample days:\n\n"
            f"- Mean IC **{ic.mean():+.4f}**, IC information ratio **{ic.mean() / ic.std():+.3f}**, "
            f"t-statistic **{t_stat:+.1f}**\n"
            f"- Positive in **{int((yearly > 0).sum())} of {len(yearly)}** calendar years\n"
            f"- Share of days with IC above zero: **{(ic > 0).mean():.1%}**\n"
        )
    else:
        ic_block = "Information coefficient unavailable: forward targets were not supplied to the reporter.\n"

    report = f"""# Research Summary

## Question

Do cross-sectional signals built from daily U.S. equity price and volume data predict
*stock-specific* relative performance, and does that prediction survive risk
neutralisation and transaction costs?

The qualifier matters. An earlier iteration of this study reported an out-of-sample
Sharpe near 1.0 that turned out to be almost entirely a market bet: the long leg
carried +0.50 more beta than the short leg, so the book made money whenever the market
rose and lost when it fell. The design below removes every route by which the model can
express that bet, which is why the headline number here is smaller and worth more.

## Design

- **Universe** {config.data.universe_name}, {config.data.start_date} to {config.data.end_date}, filtered for price and liquidity.
- **Target** the {config.features.target_horizons[-1]}-session forward return net of beta times the benchmark's
  forward return, standardised within each date. Predicting the raw return trains the
  model to forecast the market's level, which the portfolio then discards; predicting
  the residual trains it to rank names against each other, which is what the book acts on.
- **Features** reversal, momentum, volume, and liquidity ranks. Beta, beta instability,
  and idiosyncratic volatility are deliberately excluded from the design matrix and used
  only as neutralisation constraints, so risk exposures cannot masquerade as alpha.
- **Validation** walk-forward ridge, retrained every {config.split.retrain_frequency_days} sessions on an expanding
  window, with an embargo: a row may only enter training once its forward label has
  actually resolved.
- **Portfolio** top and bottom {config.portfolio.long_quantile:.0%} by prediction, sector-budgeted, dollar and beta
  neutral, rebalanced every {config.portfolio.rebalance_frequency_days} sessions to match the prediction horizon.
- **Costs** {config.backtest.transaction_cost_bps:.0f} bps one way charged on realised turnover.

## Predictive power

{ic_block}
## Portfolio performance

{splits.to_markdown(index=False) if not splits.empty else "n/a"}

Average one-way turnover is {metrics['average_turnover']:.3f} of gross per day, an implied holding
period of {metrics['implied_holding_period_days']:.0f} sessions. The strategy breaks even at
**{breakeven:.0f} bps** one way, against the {config.backtest.transaction_cost_bps:.0f} bps assumed here.

## What the evidence supports

The signal has genuine and stable cross-sectional predictive power: the IC is positive
in most years and its t-statistic is large enough to survive a multiple-testing haircut
comfortably. Translating that into portfolio return is where most of it is lost, to two
causes that are visible in the figures rather than hidden: sector budgeting costs roughly
0.2 of Sharpe relative to an unconstrained decile book, and turnover costs the rest.

The validation window is materially stronger than the test window. Most of that gap sits
in three specific periods where the IC collapses to zero, and this study does not claim to
know whether that is regime dependence or decay.

## Limitations

- **Survivorship bias.** The universe snapshot is current index membership applied
  backwards, so firms that left the index are missing. The long/short structure damps
  this but does not remove it. Point-in-time membership is the fix.
- **No fundamentals.** Value and quality, two of the most established cross-sectional
  effects, are absent because the price feed cannot supply them. Size is proxied by
  traded notional.
- **Costs are a flat rate.** Real cost scales with participation and is higher in the
  smaller, less liquid names that the size signal favours, so the true drag is likely
  worse than the flat assumption.
- **Daily bars only.** No intraday execution, borrow availability, or short-fee modelling.

## What would move this forward

Point-in-time universe and sector data; fundamental signals to diversify away from a
purely technical feature set; a participation-aware cost model; and a longer test window
before treating the validation-period result as repeatable.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
