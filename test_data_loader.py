import unittest
from unittest.mock import patch

import pandas as pd

from data_loader import load_prices


class PriceShapeTests(unittest.TestCase):
    def test_single_ticker_multiindex_download_returns_flat_ticker_columns(self):
        dates = pd.date_range("2025-01-01", periods=2)
        raw = pd.DataFrame([[100., 99.], [101., 100.]], index=dates,
                           columns=pd.MultiIndex.from_tuples(
                               [("Close", "AAPL"), ("Open", "AAPL")],
                               names=["Price", "Ticker"]))
        with patch("data_loader.yf.download", return_value=raw):
            prices = load_prices("AAPL", period="1y")
        self.assertNotIsInstance(prices.columns, pd.MultiIndex)
        pd.testing.assert_series_equal(prices["AAPL"],
                                       pd.Series([100., 101.], index=dates, name="AAPL"))

    def test_legacy_single_ticker_download_remains_supported(self):
        raw = pd.DataFrame({"Close": [100., 101.]})
        with patch("data_loader.yf.download", return_value=raw):
            prices = load_prices(["AAPL"], period="1y")
        self.assertEqual(prices.columns.tolist(), ["AAPL"])
        self.assertEqual(prices["AAPL"].tolist(), [100., 101.])

    def test_multiple_ticker_download_remains_supported(self):
        raw = pd.DataFrame([[100., 200.]], columns=pd.MultiIndex.from_tuples(
            [("AAPL", "Close"), ("MSFT", "Close")]))
        with patch("data_loader.yf.download", return_value=raw):
            prices = load_prices(["AAPL", "MSFT"], period="1y")
        self.assertEqual(prices.columns.tolist(), ["AAPL", "MSFT"])
        self.assertEqual(prices.iloc[0].tolist(), [100., 200.])
