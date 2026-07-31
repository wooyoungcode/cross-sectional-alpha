# Cross-Sectional Equity Alpha Research

A walk-forward research platform for systematic equity alpha: builds a daily
cross-sectional panel over the S&P Composite 1500, engineers price and volume
signals, produces strictly out-of-sample predictions with a label-aware training
embargo, and constructs a sector- and beta-neutral long/short book evaluated
under explicit transaction costs.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/wooyoungcode/alpha-research/actions/workflows/tests.yml/badge.svg)

---

## Headline results

Out-of-sample, 2016-2024. 1,008 names, 2,262 trading days, 21-session holding period.

| | gross | net @ 5 bps | net @ 10 bps |
|---|---|---|---|
| **Sharpe** | **0.91** | **0.82** | **0.75** |
| Annualised return | 7.45% | 6.68% | 5.92% |

| | |
|---|---|
| Information coefficient | **+0.0307**, t = **+13.9** |
| Max drawdown | **5.7%** |
| Breakeven transaction cost | **50 bps** one way, against 10 assumed |
| Average one-way turnover | 5.7% of gross per day |
| Implied holding period | 17.5 sessions |
| Realised portfolio beta | 9e-18 |

Validation (2016-2019) and test (2020-2024) both return **0.73 net at 10 bps**.
They began at 1.57 and 0.66; the convergence is the point, and the feature
neutralisation section below explains what closed it.

![Information coefficient](outputs/figures/information_coefficient.png)

---

## What this is not

The first version of this repository reported a Sharpe of 0.69. That number was
an artifact of a one-day return alignment error: weights formed on date `t` were
credited with the return from `t-1` to `t`, a move already realised before the
book existed. Measured directly, the mis-aligned series earned **-50.88 bps/day**
against **+0.05 bps/day** for the correctly aligned one. Running the same code on
503 names instead of 57 sent the reported Sharpe to **-3.53**, which is the tell:
a real edge does not invert when you add names.

Three further defects were found and fixed:

| defect | consequence |
|---|---|
| No training embargo | Rows entered training whose forward label resolved after the retrain date |
| Raw-return target | The model learned a market bet: the long leg carried **+0.50** more beta than the short leg |
| Sign-flipped weights | 20% of long-book names were held short after the neutralisation projection |
| Free opening trade | `sum` over an all-NaN `diff` row returns 0.0, so the initial book's turnover was never charged |

The headline above is the first number in this project that survives its own
diagnostics.

---

## Method

**Universe.** S&P Composite 1500, 2004-2024, filtered on price and median dollar
volume. 1,008 names clear the history requirement.

**Target.** The 21-session forward return, net of beta times the benchmark's
forward return, standardised within each date. Both choices are deliberate:

- *Beta-adjusted*, because regressing on the raw forward return rewards the model
  for loading on beta whenever the training window happens to be a rising market.
  That is a market bet dressed as alpha, and it is what the original version did.
- *Standardised within date*, because the portfolio only ever acts on the ordering
  of names on a given day. Training the model to forecast the level of returns
  spends capacity on a quantity the strategy discards.

**Features.** 23 cross-sectional price and volume signals: momentum at 12-1, 6-1,
60 and 20 sessions, reversal at 1, 5 and 21, liquidity and turnover, realised
volatility, skewness, 52-week-high proximity, Amihud illiquidity, and lottery
demand. Beta, beta instability and idiosyncratic volatility are computed but
**deliberately excluded from the design matrix** — they belong in the
neutralisation constraints, not in the signal.

Note `momentum_60d`: a 60-session window sits inside the short-term reversal
region, and its standalone rank IC against forward residual returns is *negative*
(t = -3.0). It is a reversal signal despite the name.

**Model.** Gradient-boosted trees, retrained every 21 sessions on an expanding
window, with two protections:

- *Embargo*: a row may only enter training once its forward label has actually
  resolved. Without it the model trains on labels that were unknowable at fit time.
- *Stride*: consecutive daily observations of a 21-session forward return overlap
  in 20 of their 21 days. Training on all of them inflates the apparent sample
  size without adding information, and multiplies fit cost.

**Feature neutralisation.** Half of the prediction's linear span in the features
is projected out within each date, trading raw correlation for stability. This is
the change that closed the validation-to-test gap:

| neutralisation | validation | test | gap |
|---|---|---|---|
| 0.00 | +1.57 | +0.66 | 0.91 |
| **0.50** | +1.36 | **+0.73** | **0.63** |
| 1.00 | +0.70 | +0.68 | 0.03 |

**Portfolio.** Top and bottom 20% by prediction, sector-budgeted, dollar and beta
neutral, held 21 sessions across **7 overlapping tranches** so only a fraction of
the book turns over at each rebalance. Tranching was the single largest
construction gain, taking validation net Sharpe from +0.37 to +0.69 with the
signal and holding period unchanged: it removes the dependence on which day the
whole book happens to roll on.

---

## Figures

| | |
|---|---|
| [Information coefficient](outputs/figures/information_coefficient.png) | Daily rank IC with a 63-day mean, test period shaded |
| [IC decay](outputs/figures/ic_decay.png) | IC by forward horizon, which is what sets the rebalance cadence |
| [Decile monotonicity](outputs/figures/decile_returns.png) | Mean residual return by prediction decile |
| [Equity curve](outputs/figures/equity_curve.png) | Cumulative return, gross and net |
| [Drawdown](outputs/figures/drawdown.png) | |
| [Cost sensitivity](outputs/figures/cost_sensitivity.png) | Net Sharpe against assumed cost, breakeven marked |
| [Risk exposures](outputs/figures/risk_exposures.png) | Realised beta, net exposure and largest net sector, per rebalance |

---

## Quick start

```bash
pip install -e ".[dev]"
```

Run the full pipeline:

```bash
alpha-research run --config configs/default.toml
```

Offline smoke test, no network required:

```bash
alpha-research run --config configs/synthetic.toml
```

Tests:

```bash
pytest
```

---

## What did not work

Recorded because the failures constrain the result as much as the successes.

| change | outcome |
|---|---|
| Inverse-volatility weighting | Gross Sharpe 0.65 to 0.53; it double-counts the size signal already in the model |
| Ranking within sector | Test 0.73 to 0.61 |
| Blending gradient boosting with ridge | Combined 0.75 to 0.48; ridge is too weak to average against |
| Volatility targeting | Combined rose to 0.83 but the holdout **fell** to 0.60 |
| 4-seed ensembling | +0.01 |
| 12 extra features | IC fell |
| 5 overnight/intraday features | Both windows fell, 0.73 to 0.63 |
| Deeper or longer boosting | Validation IC fell monotonically: 0.0350 at depth 3, 0.0288 at depth 7 |

The last one matters: model capacity is not the binding constraint. These
features do not contain deeper interactions to find.

---

## Honest limitations

- **Survivorship bias.** The universe snapshot is current index membership applied
  backwards, so firms that left the index are missing. The long/short structure
  damps this but does not remove it. Point-in-time membership is the fix.
- **Specification count.** Roughly 130 configurations were evaluated reaching this
  result. Selection was made on validation with test held out, and the two windows
  agree, but a search of that size widens the honest confidence interval around
  any selected number.
- **No fundamentals.** Value and quality, two of the most established
  cross-sectional effects, are absent because the feed is price-only. Size is
  proxied by traded notional.
- **Flat transaction costs.** Real cost scales with participation and is higher in
  the smaller, less liquid names the size signal favours. Breakeven at 50 bps
  leaves wide headroom, but a participation-aware model would be more honest.
- **Breadth.** 1,008 names against the 30,000 used in the published literature.
  Breadth enters risk-adjusted return as its square root, and is the largest
  single structural gap between this and results reported at Sharpe 2+.

## Where the number sits

Published machine-learning asset pricing reports out-of-sample Sharpe from 1.35
to 3.6, on universes up to 30,000 names, with hundreds of signals including
fundamentals, and **gross of transaction costs**. The linear baselines in that
same literature land at 0.5-0.9. Real equity market-neutral books, after costs,
run roughly 0.5-1.0.

At 0.91 gross and 0.82 net this sits with those linear baselines on 1/30th the
universe, and inside the range of live market-neutral strategies.

---

## Licence

MIT, see [LICENSE](LICENSE).
