# Momentum Strategy

A quality-filtered moving-average crossover strategy: generates BUY signals
on a bullish 5/20-day MA crossover only when the underlying momentum is
"HIGH quality" (smooth, steady gains) rather than a jumpy spike, and SELL
signals on a bearish crossover.

## Project structure

```
momentum_quality.py   # classifies momentum as HIGH/LOW quality over rolling windows
momentum_strategy.py  # generates BUY/SELL signals (crossover + quality filter)
data_loader.py         # pulls real price history from Yahoo Finance (yfinance)
backtest.py             # simulates the strategy and compares vs buy & hold
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Pull real data:**
```bash
python data_loader.py
# saves prices.csv for a default ticker list, or use programmatically:
```
```python
from data_loader import load_prices
prices = load_prices(["AAPL", "MSFT", "GOOGL"], period="3y")
```

**2. Get signals for a single date:**
```python
from momentum_strategy import momentum_signals
signals = momentum_signals(prices, prices.index[-1])
# [{'ticker': 'AAPL', 'signal': 'BUY', 'quality': 'HIGH', 'timestamp': ...}]
```

**3. Backtest and compare against buy & hold:**
```bash
python backtest.py
```
```python
from backtest import run_backtest
result = run_backtest(prices)
print(result["metrics"]["strategy"])
print(result["metrics"]["benchmark"])
```

Metrics reported: total return, CAGR, annualized volatility, Sharpe ratio,
max drawdown.

## Why results may still be weak (read this before tuning blindly)

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

## Notes

- `yfinance` pulls from Yahoo Finance's public endpoints — no API key
  needed, but data can occasionally have gaps or adjustments change
  retroactively. Not suitable as a system-of-record data source for live
  trading.
- `backtest.py` is for quick iteration/sanity-checking, not a production
  execution engine.
