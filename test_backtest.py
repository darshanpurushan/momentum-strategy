"""Deterministic metric regressions; no market-data downloads required."""

import unittest

import numpy as np
import pandas as pd

from backtest import _performance_metrics


class PerformanceMetricsTests(unittest.TestCase):
    def test_one_trading_year_cagr_equals_total_return(self):
        # 253 closing values span exactly 252 daily return intervals.
        for growth in (1.10, 0.80, 1.0):
            with self.subTest(growth=growth):
                curve = pd.Series(np.geomspace(100.0, 100.0 * growth, 253))
                metrics = _performance_metrics(curve)
                self.assertEqual(metrics["cagr"], round(growth - 1, 4))
                self.assertEqual(metrics["cagr"], metrics["total_return"])

    def test_two_observations_span_one_return_interval(self):
        metrics = _performance_metrics(pd.Series([100.0, 100.1]))
        self.assertEqual(metrics["cagr"], round(1.001 ** 252 - 1, 4))

    def test_single_observation_has_no_annualized_growth(self):
        metrics = _performance_metrics(pd.Series([100.0]))
        self.assertIsNone(metrics["cagr"])
        self.assertEqual(metrics["total_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
