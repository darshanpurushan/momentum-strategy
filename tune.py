"""
Parameter sweep for momentum_strategy: tests combinations of MA windows and
quality-filter strictness, backtests each, and ranks by Sharpe ratio.

IMPORTANT -- overfitting warning: this searches for the best-performing
parameters on ONE historical period and ticker list. Whatever wins here is
very likely partly fit to the specific 3-year mega-cap-tech rally in this
dataset, not a universal truth about momentum strategies. Before trusting a
"winning" combo:
  1. Re-run it on a different time period (e.g. 2018-2021, which includes
     a real crash) and a different/broader ticker list.
  2. Be suspicious of combos that only work in a narrow parameter range --
     that's a classic overfitting signature. Prefer combos where nearby
     parameter values also perform reasonably (a "plateau", not a "spike").
"""

import itertools

import momentum_strategy
from backtest import run_backtest
from data_loader import load_prices

# grid to search -- edit these to widen/narrow the search
SHORT_LONG_COMBOS = [
    (10, 30),
    (20, 50),
    (20, 100),
    (50, 200),
]
QUALITY_SCORE_THRESHOLDS = [0.0, 0.5, 1.0, 1.5]
MIN_PCT_POSITIVE_DAYS_OPTIONS = [0.40, 0.50]


def evaluation_start_index(quality_window_days=63):
    return max(max(long for _, long in SHORT_LONG_COMBOS), quality_window_days) + 1


def sweep(prices, quality_window_label="3m", quality_window_days=63):
    """
    Runs the backtest across the full parameter grid, temporarily overriding
    momentum_strategy's module-level settings for each combination.

    Returns
    -------
    list[dict], sorted by strategy Sharpe descending. Each entry has the
    parameter combo plus its resulting metrics.
    """
    start_index = evaluation_start_index(quality_window_days)
    if len(prices) <= start_index:
        raise ValueError("Not enough prices for a common evaluation window")
    results = []

    # save originals so we can restore them after the sweep
    orig = {
        "SHORT_WINDOW": momentum_strategy.SHORT_WINDOW,
        "LONG_WINDOW": momentum_strategy.LONG_WINDOW,
        "QUALITY_WINDOW_LABEL": momentum_strategy.QUALITY_WINDOW_LABEL,
        "QUALITY_WINDOW_DAYS": momentum_strategy.QUALITY_WINDOW_DAYS,
        "QUALITY_SCORE_THRESHOLD": momentum_strategy.QUALITY_SCORE_THRESHOLD,
        "MIN_PCT_POSITIVE_DAYS": momentum_strategy.MIN_PCT_POSITIVE_DAYS,
    }

    try:
        combos = itertools.product(
            SHORT_LONG_COMBOS, QUALITY_SCORE_THRESHOLDS, MIN_PCT_POSITIVE_DAYS_OPTIONS
        )
        for (short_w, long_w), q_thresh, min_pos in combos:
            momentum_strategy.SHORT_WINDOW = short_w
            momentum_strategy.LONG_WINDOW = long_w
            momentum_strategy.QUALITY_WINDOW_LABEL = quality_window_label
            momentum_strategy.QUALITY_WINDOW_DAYS = quality_window_days
            momentum_strategy.QUALITY_SCORE_THRESHOLD = q_thresh
            momentum_strategy.MIN_PCT_POSITIVE_DAYS = min_pos

            result = run_backtest(prices, start_index=start_index)
            m = result["metrics"]["strategy"]

            results.append(
                {
                    "short_window": short_w,
                    "long_window": long_w,
                    "quality_score_threshold": q_thresh,
                    "min_pct_positive_days": min_pos,
                    "total_return": m["total_return"],
                    "cagr": m["cagr"],
                    "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"],
                    "trades": len(result["trades"]),
                }
            )
    finally:
        # always restore, even if the sweep errors partway through
        for k, v in orig.items():
            setattr(momentum_strategy, k, v)

    results.sort(key=lambda r: (r["sharpe"] is not None, r["sharpe"]), reverse=True)
    return results


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    prices = load_prices(tickers, period="3y")

    results = sweep(prices)

    benchmark = run_backtest(prices, start_index=evaluation_start_index())["metrics"]["benchmark"]
    print(f"Benchmark: sharpe={benchmark['sharpe']}  total_return={benchmark['total_return']}\n")

    print(f"{'short':>6} {'long':>6} {'q_thresh':>9} {'min_pos':>8} "
          f"{'sharpe':>8} {'return':>8} {'drawdown':>9} {'trades':>7}")
    for r in results[:15]:
        print(
            f"{r['short_window']:>6} {r['long_window']:>6} "
            f"{r['quality_score_threshold']:>9} {r['min_pct_positive_days']:>8} "
            f"{r['sharpe']:>8} {r['total_return']:>8} {r['max_drawdown']:>9} {r['trades']:>7}"
        )
