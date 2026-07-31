# Research Summary

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

- **Universe** sp1500, 2004-01-01 to 2024-12-31, filtered for price and liquidity.
- **Target** the 21-session forward return net of beta times the benchmark's
  forward return, standardised within each date. Predicting the raw return trains the
  model to forecast the market's level, which the portfolio then discards; predicting
  the residual trains it to rank names against each other, which is what the book acts on.
- **Features** reversal, momentum, volume, and liquidity ranks. Beta, beta instability,
  and idiosyncratic volatility are deliberately excluded from the design matrix and used
  only as neutralisation constraints, so risk exposures cannot masquerade as alpha.
- **Validation** walk-forward ridge, retrained every 21 sessions on an expanding
  window, with an embargo: a row may only enter training once its forward label has
  actually resolved.
- **Portfolio** top and bottom 20% by prediction, sector-budgeted, dollar and beta
  neutral, rebalanced every 21 sessions to match the prediction horizon.
- **Costs** 10 bps one way charged on realised turnover.

## Predictive power

Rank IC against the 21-session beta-residual forward return, measured on 2,242 out-of-sample days:

- Mean IC **+0.0307**, IC information ratio **+0.294**, t-statistic **+13.9**
- Positive in **9 of 9** calendar years
- Share of days with IC above zero: **62.4%**

## Portfolio performance

| split      |   days | annualised return   | annualised vol   |   Sharpe |   Newey-West t | max drawdown   |
|:-----------|-------:|:--------------------|:-----------------|---------:|---------------:|:---------------|
| validation |   1005 | +3.49%              | 2.52%            |     1.36 |           2.74 | -3.0%          |
| test       |   1257 | +7.90%              | 10.38%           |     0.73 |           1.67 | -5.6%          |
| combined   |   2262 | +5.92%              | 7.92%            |     0.73 |           2.21 | -5.6%          |

Average one-way turnover is 0.057 of gross per day, an implied holding
period of 18 sessions. The strategy breaks even at
**50 bps** one way, against the 10 bps assumed here.

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
