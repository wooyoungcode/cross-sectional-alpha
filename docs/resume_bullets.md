# Resume Bullets

Two bullets, sized for a resume line. Every figure is reproducible from
`configs/default.toml` and appears in the README.

---

## Preferred pair

> Built a walk-forward cross-sectional equity alpha platform over 1,000 S&P 1500
> names (2004-2024) with a label-aware training embargo, beta-residual targets, and
> sector- and beta-neutral construction across 7 overlapping tranches, reaching an
> out-of-sample information coefficient of 0.031 (t = 13.9) and a Sharpe of 1.22
> gross, 1.00 net of realistic transaction costs, on 2.8% annualised return at
> 3.2% volatility and a 6.4% maximum drawdown.

> Diagnosed the original backtest's reported Sharpe as an artifact of a one-day
> return-alignment error worth -51 bps/day and an unintended +0.50 beta tilt;
> rebuilt the target and neutralisation so the delivered book is beta-flat to 1e-17,
> and filtered 129 corporate-action artifacts across 54 tickers, one of which
> alone moved the book 21.9% in a single day.

---

## Why these two

The first is the result. It leads with the information coefficient and its
t-statistic rather than the Sharpe, because a t of 13.9 over 2,262 days is the
harder number to dismiss, and it quotes gross *and* net so the figure survives the
inevitable "gross or net?" follow-up.

The second is the differentiator. Most candidates present a backtest; far fewer can
show they found their own result was wrong and quantified why. It also sets up the
strongest talking point available here: the reported 0.69 was a bug, and the tell
was that expanding the universe from 57 names to 503 flipped it to -3.53.

---

## Interview follow-ups this invites, and the answers

**"How many configurations did you try?"** Roughly 130. Selection was on the
validation split with test held out: validation returns 1.00 net at 10 bps and
test 0.62. The two windows once stood at 1.57 and 0.66; feature neutralisation at
0.5 closed most of that gap by trading raw correlation for stability.

**"Gross or net?"** Both are in the README. 1.22 gross, 1.00 at 5 bps, 0.78 at 10.
Breakeven is 28 bps against 10 assumed, so the result is not a cost-assumption
artifact. Say the return alongside it: 2.8% a year at 3.2% volatility. This is a
low-return, low-volatility book, and quoting the ratio alone invites a question
better answered upfront.

**"Why is it not higher?"** Breadth and data. The published 1.35 to 3.6 figures use
up to 30,000 names and hundreds of signals including fundamentals, gross of costs;
the linear baselines in those same papers sit at 0.5 to 0.9. Breadth enters
risk-adjusted return as its square root, and this universe is 1/30th the size.

**"What did you try that failed?"** Eight things, tabulated in the README with their
numbers. Volatility targeting is the instructive one: it raised the combined Sharpe
to 0.83 while the untouched holdout fell to 0.60, so it was rejected.

**"How do you know there is no look-ahead?"** The embargo is asserted directly in
`tests/test_modeling.py` against the training-selection rule, rather than against
downstream metrics, which can absorb a leak without failing.
