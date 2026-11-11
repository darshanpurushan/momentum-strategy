import numpy as np
import pandas as pd

from momentum_strategy import momentum_signals


def test_gap_inside_moving_average_window_cannot_trigger_exit():
    prices = pd.DataFrame({'A': np.arange(70., 0., -1.)})
    prices.loc[50, 'A'] = np.nan
    assert momentum_signals(prices, 69) == []


def test_old_gap_does_not_prevent_exit_after_complete_warmup():
    prices = pd.DataFrame({'A': np.arange(100., 0., -1.)})
    prices.loc[0, 'A'] = np.nan
    signals = momentum_signals(prices, 99)
    assert [(s['ticker'], s['signal']) for s in signals] == [('A', 'SELL')]
