"""
Out-of-sample validation: runs the (now locked-in) momentum strategy across
several distinct historical periods -- not just the recent bull run used
for tuning -- to check whether it holds up across different market regimes
(crashes, choppy/sideways periods, different eras), not just the one window
its parameters were picked on.

This is the real test of whether tune.py's parameter choice was genuine
signal or just fit to one dataset.
"""

import pandas as pd

from backtest import run_backtest
from data_loader import load_prices

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]

# distinct regimes worth checking:
#  - a calmer pre-pandemic bull market
#  - the 2018 selloff + recovery (a real, if brief, crash)
#  - the COVID crash and V-shaped recovery (extreme volatility)
#  - the 2022 bear market + 2023 recovery (sustained drawdown, different
#    macro regime -- rate hikes -- than the other periods)
PERIODS = {
    "2015-2018 (pre-pandemic bull)": ("2015-01-01", "2018-01-01"),
    "2018-2020 (2018 selloff + recovery)": ("2018-01-01", "2020-01-01"),
    "2020-2022 (COVID crash + recovery)": ("2020-01-01", "2022-01-01"),
    "2022-2024 (2022 bear market + 2023 recovery)": ("2022-01-01", "2024-01-01"),
}


def validate():
    rows = []

    for label, (start, end) in PERIODS.items():
        prices = load_prices(TICKERS, start=start, end=end)

        if len(prices) < 60:
            print(f"Skipping {label}: not enough data returned ({len(prices)} rows)")
            continue

        result = run_backtest(prices)
        strat = result["metrics"]["strategy"]
        bench = result["metrics"]["benchmark"]

        rows.append(
            {
                "period": label,
                "strategy_return": strat["total_return"],
                "strategy_sharpe": strat["sharpe"],
                "strategy_max_dd": strat["max_drawdown"],
                "benchmark_return": bench["total_return"],
                "benchmark_sharpe": bench["sharpe"],
                "benchmark_max_dd": bench["max_drawdown"],
                "trades": len(result["trades"]),
            }
        )

    return pd.DataFrame(rows).set_index("period")


if __name__ == "__main__":
    results = validate()
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print(results)

    print("\n--- What to look for ---")
    print("1. Does strategy_sharpe beat benchmark_sharpe in the crash/choppy")
    print("   periods (2018, 2020, 2022), even if it lags in calm bull runs?")
    print("   A trend-following + quality filter strategy SHOULD shine when")
    print("   the market isn't just going straight up.")
    print("2. Is strategy_max_dd meaningfully shallower than benchmark_max_dd")
    print("   during the crash periods? That's the actual value proposition")
    print("   of exiting on bearish crossovers -- capital protection, not")
    print("   necessarily beating buy & hold in every regime.")