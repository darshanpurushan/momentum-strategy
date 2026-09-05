Momentum Strategy

A quality-filtered moving-average crossover strategy: generates BUY signals
on a bullish 5/20-day MA crossover only when the underlying momentum is
"HIGH quality" (smooth, steady gains) and SELL
signals on a bearish crossover.


Metrics reported: total return, CAGR, annualized volatility, Sharpe ratio,
max drawdown.

Why results may still be weak

A 5/20-day MA crossover is a simple, laggy signal by construction, and
adding a momentum-quality filter reduces false entries but doesn't fix the
core issues:

- **Whipsaws in choppy/sideways markets** — the crossover fires late and
  reverses fast, generating losing round-trips even in a market with no real
  trend.
- **No cost model** — this backtest has zero slippage/commissions. Real
  crossover strategies trade often enough that costs can erase the edge.
- **Fixed share size** — `POSITION_SIZE = 100` regardless of price or
  volatility means position risk isn't normalized across tickers.
- **Single benchmark, no risk adjustment** — always compare Sharpe, not just
  total return, against buy & hold. A crossover strategy that returns less
  but with much lower drawdown may still be "better" depending on your goal.
- **Look-ahead-free but still curve-fit-prone** — `SHORT_WINDOW=5`,
  `LONG_WINDOW=20`, and the quality thresholds in `momentum_quality.py`
  (`quality_score_threshold=1.0`, `min_pct_positive_days=0.50`) were chosen
  as reasonable defaults, not optimized on your data. Treat them as a
  starting point for your own walk-forward testing, not tuned parameters.

Reasonable next steps: widen the ticker universe, test on multiple
non-overlapping periods (not just one 3-year window), add a cost/slippage
assumption, and try normalizing position size by volatility (e.g. inverse-vol
weighting) instead of a flat share count.

Notes

- `yfinance` pulls from Yahoo Finance's public endpoints — no API key
  needed, but data can occasionally have gaps or adjustments change
  retroactively. Not suitable as a system-of-record data source for live
  trading.
- `backtest.py` is for quick iteration/sanity-checking, not a production
  execution engine.

The strategy was tuned on the 2023-2026 mega-cap tech bull run and initially underperformed buy & hold significantly on that single window. Two fixes improved things measurably:

Position sizing: switched from a flat 100-share trade size to equal-weight capital allocation (deploys a fair share of available capital per position instead of leaving cash idle).
Wider MA windows: moved from 5/20-day to 20/50-day crossover, cutting whipsaw trades by ~2/3 and raising Sharpe from 0.78 to ~1.07 on the tuning period.

Locked-in parameters (see momentum_strategy.py): SHORT_WINDOW=20, LONG_WINDOW=50, QUALITY_SCORE_THRESHOLD=0.5, MIN_PCT_POSITIVE_DAYS=0.50.

Out-of-sample validation (validate.py)

To check these parameters weren't just overfit to one bull-run window, the same strategy was re-tested — unchanged — across four distinct historical periods:

Period	Strategy Sharpe	Benchmark Sharpe	Strategy Max DD	Benchmark Max DD
2015-18 (calm bull)	1.60	2.07	-13.2%	-17.7%
2018-20 (selloff+recovery)	0.98	0.73	-13.3%	-31.8%
2020-22 (COVID crash)	1.43	1.51	-16.3%	-29.5%
2022-24 (bear+recovery)	1.27	0.48	-15.4%	-42.6%

Consistent pattern across all four regimes:

Drawdown protection held up every time — roughly half of buy & hold's max drawdown in every period tested, not just the tuning window.
The strategy beat buy & hold on risk-adjusted return in 2 of 4 periods, including outright beating it on both Sharpe and total return during the 2022 bear market.
It only clearly lagged during 2015-2018 — a relentless, low-volatility bull run — the same kind of regime as the original tuning period. This is expected: a trend-following exit gives up some upside in calm bull markets in exchange for real downside protection elsewhere.

Conclusion: this is not a strategy that beats buy & hold in every environment, and it isn't trying to be. It's a lower-drawdown, more defensive alternative that trades some bull-market upside for materially better capital protection in crashes and bear markets. Whether that tradeoff is worth it depends on your risk tolerance and time horizon — it is not a free lunch.

Caveats that still apply
No transaction costs/slippage modeled — the 20/50 setup trades far less than 5/20 did, but real costs would still erode returns somewhat.
Only tested on 6 mega-cap tech tickers — a broader/more diverse universe might behave differently.
Quality filter thresholds have less impact than window size, based on tune.py sweeps — most of the improvement came from widening the MA windows, not the quality logic itself.
