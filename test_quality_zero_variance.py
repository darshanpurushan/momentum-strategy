import unittest

import numpy as np
import pandas as pd

from momentum_quality import classify_momentum_quality, rolling_momentum_quality


class ZeroVarianceQualityTests(unittest.TestCase):
    def test_constant_returns_have_undefined_quality_in_both_interfaces(self):
        # Powers of two give exactly representable returns with zero variance.
        for base in (2., .5, 1.):
            with self.subTest(base=base):
                prices = pd.DataFrame({"AAPL": base ** np.arange(10)})
                snapshot = classify_momentum_quality(prices, "AAPL", windows={"test": 3})
                rolling = rolling_momentum_quality(prices, "AAPL", windows={"test": 3})["test"]
                self.assertTrue(pd.isna(snapshot.iloc[-1]["ann_quality_score"]))
                self.assertTrue(pd.isna(rolling.iloc[-1]["ann_quality_score"]))
                self.assertTrue(pd.isna(rolling.iloc[-1]["quality"]))

    def test_varying_returns_match_snapshot_quality(self):
        prices = pd.DataFrame({"AAPL": [100., 102., 101., 104., 106., 105.]})
        snapshot = classify_momentum_quality(prices, "AAPL", windows={"test": 3})
        rolling = rolling_momentum_quality(prices, "AAPL", windows={"test": 3})["test"]
        self.assertAlmostEqual(snapshot.iloc[-1]["ann_quality_score"],
                               rolling.iloc[-1]["ann_quality_score"])
        self.assertEqual(snapshot.iloc[-1]["quality"], rolling.iloc[-1]["quality"])
