import unittest

import numpy as np
import pandas as pd

from momentum_quality import classify_momentum_quality


class QualityWindowTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame({"AAPL": 100. + np.arange(80) ** 1.1})

    def test_mixed_standard_and_custom_windows_are_all_returned(self):
        out = classify_momentum_quality(self.prices, "AAPL",
                                       windows={"10d": 10, "2m": 42, "1m": 21, "5d": 5})
        self.assertEqual(out.index.tolist(), ["1m", "2m", "10d", "5d"])
        self.assertEqual(out.loc["10d", "window_days"], 10)
        expected = self.prices["AAPL"].iloc[-1] / self.prices["AAPL"].iloc[-11] - 1
        self.assertAlmostEqual(out.loc["10d", "momentum_return"], expected)

    def test_custom_only_windows_preserve_requested_order(self):
        out = classify_momentum_quality(self.prices, "AAPL", windows={"10d": 10, "5d": 5})
        self.assertEqual(out.index.tolist(), ["10d", "5d"])

    def test_short_history_still_includes_custom_window(self):
        out = classify_momentum_quality(self.prices.head(4), "AAPL",
                                       windows={"1m": 21, "10d": 10})
        self.assertIn("10d", out.index)
        self.assertTrue(pd.isna(out.loc["10d", "quality"]))
