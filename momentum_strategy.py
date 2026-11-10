"""
Momentum crossover signal generator, filtered by momentum quality.

BUY signal: bullish 20/50-day MA crossover AND HIGH quality recent momentum
(smooth, not driven by a few big jumps).
SELL signal: bearish 20/50-day MA crossover (quality is not checked on exits
-- you always want a clean way out when the trend turns).

This module only *generates* signals as a list -- it does not place trades.
Wire it into `backtest.py` (included in this project) or your own execution
engine.
"""

import pandas as pd

from momentum_quality import classify_momentum_quality

# 20/50-day is a more standard trend-following pairing than 5/20. The
# original 5/20 combo reacted to nearly every short-term wiggle, generating
# very frequent trades (whipsaws) that underperformed buy & hold even after
# fixing position sizing. Widening the windows makes the signal react to the
# underlying trend rather than daily noise, at the cost of being slower to
# both enter and exit.
SHORT_WINDOW = 20
LONG_WINDOW = 50

# which momentum_quality lookback to use as the entry filter -- widened from
# "1m" to "3m" to match the longer trend horizon above
QUALITY_WINDOW_LABEL = "3m"
QUALITY_WINDOW_DAYS = 63

# Quality filter strictness. Higher QUALITY_SCORE_THRESHOLD and
# MIN_PCT_POSITIVE_DAYS = fewer, "smoother" entries allowed through; lower
# values let in more entries, including volatile-but-still-uptrending names
# (relevant for names like NVDA that trend up but with sharp swings).
QUALITY_SCORE_THRESHOLD = 0.5
MIN_PCT_POSITIVE_DAYS = 0.50


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

    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price index must be unique and increasing")
    if timestamp not in prices.index:
        raise ValueError("timestamp must identify an observed price bar")
    history = prices.loc[:timestamp]
    if len(history) < LONG_WINDOW + 1:
        return signals

    for ticker in prices.columns:
        if pd.isna(history[ticker].iloc[-1]):
            continue
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
                history,
                ticker,
                windows={QUALITY_WINDOW_LABEL: QUALITY_WINDOW_DAYS},
                quality_score_threshold=QUALITY_SCORE_THRESHOLD,
                min_pct_positive_days=MIN_PCT_POSITIVE_DAYS,
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
