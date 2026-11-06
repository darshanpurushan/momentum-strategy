import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest import run_backtest


class BacktestPriceValidationTests(unittest.TestCase):
    def test_invalid_prices_are_rejected_before_signal_evaluation(self):
        for value in (np.nan, np.inf, -np.inf, 0., -1.):
            with self.subTest(value=value):
                prices = pd.DataFrame({"AAPL": [100., value, 102.]})
                with patch("backtest.momentum_signals", return_value=[]) as signals:
                    with self.assertRaisesRegex(ValueError, "finite positive prices"):
                        run_backtest(prices, start_index=0)
                    signals.assert_not_called()

    def test_empty_simulation_inputs_have_clear_errors(self):
        for prices, start in ((pd.DataFrame(), 0),
                              (pd.DataFrame(index=range(3)), 0),
                              (pd.DataFrame({"AAPL": [100., 101.]}), 2),
                              (pd.DataFrame({"AAPL": [100., 101.]}), -1)):
            with self.subTest(shape=prices.shape, start=start):
                with self.assertRaises(ValueError):
                    run_backtest(prices, start_index=start)

    def test_valid_prices_preserve_cash_when_no_signals_fire(self):
        prices = pd.DataFrame({"AAPL": [100., 101., 102.]})
        with patch("backtest.momentum_signals", return_value=[]):
            result = run_backtest(prices, start_index=0)
        self.assertEqual(result["equity_curve"].tolist(), [100000.] * 3)
        self.assertEqual(result["trades"], [])
