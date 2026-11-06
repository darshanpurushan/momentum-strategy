"""
Momentum quality classifier.

Given a DataFrame of prices (columns = tickers, index = dates) and a ticker,
computes momentum over several rolling lookback windows and classifies each
window's momentum as HIGH or LOW quality.

Quality logic:
    - HIGH quality = the return was earned smoothly (steady daily gains,
      low volatility relative to the return -> high "Sharpe-like" score).
    - LOW quality  = the return was driven by a few large jumps (high
      volatility relative to the return, or a low % of positive days).
"""

import numpy as np
import pandas as pd

# Approximate trading-day lengths for common lookback labels
DEFAULT_WINDOWS = {
    "1m": 21,
    "2m": 42,
    "3m": 63,
    "6m": 126,
    "12m": 252,
    "18m": 378,
}


def classify_momentum_quality(
    prices: pd.DataFrame,
    ticker: str,
    windows: dict = None,
    quality_score_threshold: float = 1.0,
    min_pct_positive_days: float = 0.50,
) -> pd.DataFrame:
    """
    Compute momentum + quality classification for one ticker across multiple
    rolling lookback windows, as of the LAST date in `prices`.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    if ticker not in prices.columns:
        raise ValueError(f"'{ticker}' not found in prices columns")

    series = prices[ticker].dropna()
    daily_returns = series.pct_change()

    rows = {}
    for label, window in windows.items():
        if len(series) < window + 1:
            rows[label] = {
                "window_days": window,
                "momentum_return": np.nan,
                "ann_quality_score": np.nan,
                "pct_positive_days": np.nan,
                "quality": None,
            }
            continue

        momentum_return = series.iloc[-1] / series.iloc[-(window + 1)] - 1
        window_returns = daily_returns.iloc[-window:]

        mean_ret = window_returns.mean()
        std_ret = window_returns.std()
        ann_quality_score = (
            (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan
        )

        pct_positive_days = (window_returns > 0).mean()

        if pd.isna(ann_quality_score):
            quality = None
        elif (
            momentum_return > 0
            and ann_quality_score >= quality_score_threshold
            and pct_positive_days >= min_pct_positive_days
        ):
            quality = "HIGH"
        else:
            quality = "LOW"

        rows[label] = {
            "window_days": window,
            "momentum_return": momentum_return,
            "ann_quality_score": ann_quality_score,
            "pct_positive_days": pct_positive_days,
            "quality": quality,
        }

    result = pd.DataFrame(rows).T
    ordered_labels = [l for l in DEFAULT_WINDOWS if l in result.index]
    result = result.loc[ordered_labels] if ordered_labels else result
    return result


def rolling_momentum_quality(
    prices: pd.DataFrame,
    ticker: str,
    windows: dict = None,
    quality_score_threshold: float = 1.0,
    min_pct_positive_days: float = 0.50,
) -> dict:
    """
    Full time-series version of classify_momentum_quality: computes momentum
    and quality classification for EVERY date, for each rolling window.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    if ticker not in prices.columns:
        raise ValueError(f"'{ticker}' not found in prices columns")

    series = prices[ticker]
    daily_returns = series.pct_change()
    is_positive_day = (daily_returns > 0).astype(float)

    results = {}
    for label, window in windows.items():
        momentum_return = series.pct_change(window)

        roll_mean = daily_returns.rolling(window).mean()
        roll_std = daily_returns.rolling(window).std()
        # Match the snapshot classifier: zero variance has no defined
        # Sharpe-like quality score, rather than an infinite score.
        ann_quality_score = (roll_mean / roll_std.where(roll_std > 0)) * np.sqrt(252)

        pct_positive_days = is_positive_day.rolling(window).mean()

        quality = pd.Series(
            np.where(
                (momentum_return > 0)
                & (ann_quality_score >= quality_score_threshold)
                & (pct_positive_days >= min_pct_positive_days),
                "HIGH",
                "LOW",
            ),
            index=series.index,
        )
        not_enough_history = ann_quality_score.isna()
        quality = quality.mask(not_enough_history, None)

        results[label] = pd.DataFrame(
            {
                "momentum_return": momentum_return,
                "ann_quality_score": ann_quality_score,
                "pct_positive_days": pct_positive_days,
                "quality": quality,
            }
        )

    return results


def rolling_quality_summary(
    prices: pd.DataFrame,
    ticker: str,
    windows: dict = None,
    quality_score_threshold: float = 1.0,
    min_pct_positive_days: float = 0.50,
) -> pd.DataFrame:
    """
    Collapses rolling_momentum_quality's dict of DataFrames into a single
    DataFrame indexed by date, with one HIGH/LOW column per window label.
    """
    per_window = rolling_momentum_quality(
        prices, ticker, windows, quality_score_threshold, min_pct_positive_days
    )
    return pd.DataFrame({label: df["quality"] for label, df in per_window.items()})
