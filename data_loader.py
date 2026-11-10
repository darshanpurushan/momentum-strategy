"""
Loads historical price data from Yahoo Finance (via the `yfinance` package)
-- free, no API key required.

Returns data in the same wide-DataFrame shape used throughout this project:
index = date, columns = ticker, values = adjusted close price.
"""

import pandas as pd
import yfinance as yf


def load_prices(tickers, start=None, end=None, period=None, interval="1d") -> pd.DataFrame:
    """
    Parameters
    ----------
    tickers : str or list[str]
        e.g. "AAPL" or ["AAPL", "MSFT", "GOOGL"]
    start, end : str, optional
        'YYYY-MM-DD'. Ignored if `period` is given.
    period : str, optional
        e.g. '1y', '2y', '5y', 'max'. Takes precedence over start/end.
    interval : str, default '1d'
        Bar size, e.g. '1d', '1wk', '1mo'.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame of adjusted close prices, index = date, columns = ticker.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    kwargs = dict(interval=interval, auto_adjust=True, progress=False)
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    if len(tickers) == 1:
        raw = yf.download(tickers[0], **kwargs)
        if raw.empty:
            return pd.DataFrame(index=raw.index[:0], columns=tickers, dtype=float)
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raw = yf.download(tickers, group_by="ticker", **kwargs)
        if raw.empty:
            return pd.DataFrame(index=raw.index[:0], columns=tickers, dtype=float)
        available = [t for t in tickers if t in raw.columns.get_level_values(0)]
        missing = sorted(set(tickers) - set(available))
        if missing:
            print(f"Warning: no data returned for {missing}")
        if not available:
            return pd.DataFrame(index=raw.index[:0], columns=tickers, dtype=float)
        prices = pd.concat({t: raw[t]["Close"] for t in available}, axis=1)

    prices = prices.dropna(how="all")
    return prices


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    prices = load_prices(tickers, period="3y")
    print(prices.tail())
    prices.to_csv("prices.csv")
    print(f"\nSaved {len(prices)} rows for {len(prices.columns)} tickers to prices.csv")
