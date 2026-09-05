"""
Lightweight backtest for the `momentum_signals` generator.

Simulates a long-only, equal-weight-capital-allocation model over historical
prices, then reports performance metrics and compares against an
equal-weight buy & hold benchmark on the same tickers/period. This is for
quick sanity-checking on real data -- not a production execution engine (no
slippage, no commissions, no partial fills).
"""

import numpy as np
import pandas as pd

from momentum_strategy import momentum_signals

INITIAL_CASH = 100_000

# Fraction of CURRENT portfolio value to deploy into a new position when a
# BUY signal fires. 1 / len(tickers) means "one equal-weight slot per
# ticker" -- e.g. with 6 tickers, each new position targets ~1/6 of the
# portfolio. This replaces flat-share-count sizing, which left large amounts
# of cash sitting idle (earning 0%) while the benchmark was fully invested
# from day one -- a major, easy-to-miss source of underperformance.
def run_backtest(prices: pd.DataFrame, start_index: int = 21) -> dict:
    """
    Parameters
    ----------
    prices : pd.DataFrame
        Wide price history, index = dates, columns = tickers.
    start_index : int
        Row to start simulating from. Needs enough history behind it for
        the longest lookback used by momentum_strategy (20-day MA + 21-day
        quality window), so don't set this below ~21.

    Returns
    -------
    dict with:
        equity_curve : pd.Series, daily strategy portfolio value
        benchmark_curve : pd.Series, equal-weight buy & hold value
        trades : list[dict] of executed trades
        metrics : dict with 'strategy' and 'benchmark' performance stats
    """
    cash = INITIAL_CASH
    holdings = {}  # ticker -> shares held
    trades = []
    equity = []

    dates = prices.index[start_index:]
    target_allocation_pct = 1.0 / len(prices.columns)

    for ts in dates:
        todays_prices = prices.loc[ts]
        signals = momentum_signals(prices, ts)

        # current mark-to-market portfolio value, used to size new entries
        holdings_value = sum(
            holdings.get(t, 0) * todays_prices.get(t, 0) for t in holdings
        )
        portfolio_value = cash + holdings_value

        for sig in signals:
            ticker = sig["ticker"]
            price = todays_prices.get(ticker, np.nan)
            if pd.isna(price) or price <= 0:
                continue

            if sig["signal"] == "BUY" and ticker not in holdings:
                target_dollars = target_allocation_pct * portfolio_value
                spend = min(target_dollars, cash)
                shares = int(spend // price)

                if shares > 0:
                    cost = shares * price
                    cash -= cost
                    holdings[ticker] = shares
                    trades.append(
                        {"timestamp": ts, "ticker": ticker, "side": "BUY",
                         "price": price, "shares": shares}
                    )

            elif sig["signal"] == "SELL" and ticker in holdings:
                shares = holdings.pop(ticker)
                cash += price * shares
                trades.append(
                    {"timestamp": ts, "ticker": ticker, "side": "SELL",
                     "price": price, "shares": shares}
                )

        holdings_value = sum(
            holdings.get(t, 0) * todays_prices.get(t, 0) for t in holdings
        )
        equity.append(cash + holdings_value)

    equity_curve = pd.Series(equity, index=dates, name="strategy")

    # equal-weight buy & hold benchmark over the same window
    bench_prices = prices.loc[dates]
    normalized = bench_prices / bench_prices.iloc[0]
    benchmark_curve = (normalized.mean(axis=1) * INITIAL_CASH).rename("benchmark")

    metrics = {
        "strategy": _performance_metrics(equity_curve),
        "benchmark": _performance_metrics(benchmark_curve),
    }

    return {
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "trades": trades,
        "metrics": metrics,
    }


def _performance_metrics(equity_curve: pd.Series) -> dict:
    returns = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years = len(equity_curve) / 252
    cagr = (
        (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1
        if n_years > 0
        else np.nan
    )
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else np.nan
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4) if not np.isnan(cagr) else None,
        "annual_vol": round(vol, 4),
        "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else None,
        "max_drawdown": round(max_drawdown, 4),
    }


if __name__ == "__main__":
    from data_loader import load_prices

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    prices = load_prices(tickers, period="3y")

    result = run_backtest(prices)

    print("Strategy metrics:  ", result["metrics"]["strategy"])
    print("Benchmark metrics: ", result["metrics"]["benchmark"])
    print(f"Total trades: {len(result['trades'])}")

