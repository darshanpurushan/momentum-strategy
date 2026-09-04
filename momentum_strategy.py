"""
Momentum crossover signal generator, filtered by momentum quality.

BUY signal: bullish 5/20-day MA crossover AND HIGH quality recent momentum
(smooth, not driven by a few big jumps).
SELL signal: bearish 5/20-day MA crossover (quality is not checked on exits
-- you always want a clean way out when the trend turns).

This module only *generates* signals as a list -- it does not place trades.
Wire it into `backtest.py` (included in this project) or your own execution
engine.
"""

import pandas as pd

from momentum_quality import classify_momentum_quality

SHORT_WINDOW = 5
LONG_WINDOW = 20

# which momentum_quality lookback to use as the entry filter
QUALITY_WINDOW_LABEL = "1m"
QUALITY_WINDOW_DAYS = 21


def momentum_signals(prices: pd.DataFrame, timestamp) -> list:
    """
    Evaluate the momentum crossover + quality filter for every ticker as of
    `timestamp` and return a list of signals.

    Parameters
    ----------
    prices : pd.DataFrame
        Full wide price history, index = dates, columns = tickers.
    timestamp : the bar's timestamp to evaluate (must exist in prices.index)

    Returns
    -------
    list[dict]
        One entry per ticker with an active signal, e.g.:
        {"ticker": "AAPL", "signal": "BUY", "quality": "HIGH", "timestamp": ts}
        Tickers with no crossover are skipped. BUY signals require HIGH
        quality momentum to appear at all.
    """
    signals = []

    history = prices.loc[:timestamp]
    if len(history) < LONG_WINDOW + 1:
        return signals

    for ticker in prices.columns:
        series = history[ticker].dropna()
        if len(series) < LONG_WINDOW + 1:
            continue

        short_ma = series.rolling(SHORT_WINDOW).mean()
        long_ma = series.rolling(LONG_WINDOW).mean()

        if pd.isna(short_ma.iloc[-1]) or pd.isna(long_ma.iloc[-1]):
            continue

        bullish = short_ma.iloc[-1] > long_ma.iloc[-1]
        bearish = short_ma.iloc[-1] < long_ma.iloc[-1]

        if bullish:
            if len(series) < QUALITY_WINDOW_DAYS + 1:
                continue

            quality_result = classify_momentum_quality(
                history, ticker, windows={QUALITY_WINDOW_LABEL: QUALITY_WINDOW_DAYS}
            )
            quality = quality_result.loc[QUALITY_WINDOW_LABEL, "quality"]

            if quality == "HIGH":
                signals.append(
                    {
                        "ticker": ticker,
                        "signal": "BUY",
                        "quality": quality,
                        "timestamp": timestamp,
                    }
                )

        elif bearish:
            signals.append(
                {
                    "ticker": ticker,
                    "signal": "SELL",
                    "quality": None,
                    "timestamp": timestamp,
                }
            )

    return signals
