# Changelog

All notable changes to this project will be documented here.

## [0.2.0] - 2026-07-30

Correctness pass. The 0.1.0 backtest reported a Sharpe of 0.69 that was an
artifact; every headline number below is a replacement, not an improvement on a
comparable figure.

### Fixed
- **Return alignment.** Weights formed on date `t` were credited with the return
  from `t-1` to `t`, a move realised before the book existed. Measured directly,
  the mis-aligned series earned -50.88 bps/day against +0.05 bps/day aligned.
- **Training embargo.** Walk-forward training used every row dated before the
  retrain date, including rows whose forward label resolved afterwards. Training
  now requires the label to have resolved. Asserted in `tests/test_modeling.py`.
- **Sign-flipped weights.** The neutralisation projection could invert a name's
  position relative to the leg it was selected into: 20% of long-book names ended
  up held short. Beta neutrality is now checked for feasibility first, and the
  projection never ships a sign violation.
- **Free opening trade.** `sum` over an all-NaN `diff` row returns 0.0, so the
  fallback never fired and the initial book's turnover was never charged.
- **Silent benchmark failure.** A rate-limited yfinance request returns all-NaN
  columns rather than raising; an empty benchmark nulled every beta and residual
  target without error. Downloads are now batched with retries, and an empty
  benchmark raises.
- Three broken CI badge and clone URLs, a missing `tabulate` dependency that
  killed every report run, and the LICENCE year.

### Changed
- **Target** is the beta-residual forward return, standardised within date, at a
  21-session horizon. The previous raw-return target let the model express a
  market bet: the long leg carried +0.50 more beta than the short leg.
- **Universe** is the S&P Composite 1500 (1,008 usable names, 2004-2024), from a
  57-name mega-cap list.
- **Model** is gradient-boosted trees. Depth and iteration count were swept on
  validation; the original shallow setting won, so capacity is not the binding
  constraint.
- **Construction** uses a 20% book width across 7 overlapping tranches, the
  largest single improvement here (+0.37 to +0.69 validation net Sharpe).
- Risk features are excluded from the design matrix and used only as
  neutralisation constraints.

### Added
- `universe.py`: S&P 500 / Composite 1500 membership with an offline snapshot and
  an explicit survivorship-bias note.
- Feature neutralisation, which closed the validation-to-test Sharpe gap from
  0.91 to 0.63 while raising the untouched holdout from 0.66 to 0.73.
- Training-date striding: consecutive observations of a 21-session forward return
  overlap in 20 of 21 days, so most rows are near-duplicates.
- Eight-figure report set in one style, and a research summary carrying the
  specification count and limitations.

### Results
Out-of-sample 2016-2024: Sharpe 1.22 gross, 1.00 at 5 bps, 0.78 at 10 bps, on
2.54% annualised return at 3.2% volatility. IC +0.0307 (t = +13.9). Max drawdown
6.4%. Breakeven cost 28 bps one way.

These supersede figures reported earlier in this release cycle (5.92% return at
0.75 net), which were inflated by unfiltered corporate actions; see the
`max_abs_daily_return` filter.

## [0.1.0] - 2026-04-04

### Added
- Initial release of the cross-sectional equity alpha research platform.
- `data.py`: synthetic and yfinance panel builders with pickle caching.
- `features.py`: momentum, reversal, volatility, volume, and beta cross-sectional features with leakage-safe forward-return targets.
- `modeling.py`: walk-forward ridge regression and rank-composite signal with strict out-of-sample splits.
- `portfolio.py`: sector-neutral, beta-neutral long/short portfolio construction via least-squares constraint projection.
- `backtest.py`: daily P&L attribution, turnover, drawdown, and volatility-regime metrics.
- `reporting.py`: equity curve and leg contribution plots, research summary report generator.
- `cli.py`: `alpha-research run --config <path>` entry point.
- `configs/synthetic.toml`: offline smoke test config.
- `configs/default.toml`: live market-data config using yfinance.
- Tests for forward-target accuracy, cross-sectional z-score centering, walk-forward split boundaries, portfolio neutrality, and transaction cost mechanics.
