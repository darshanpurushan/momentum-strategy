import pandas as pd
from unittest.mock import patch

from backtest import run_backtest


def test_same_bar_exit_funds_entry_regardless_of_signal_order():
    prices = pd.DataFrame({'NEW': [10., 10., 10., 12.],
                           'OLD': [10., 20., 20., 20.]})

    def simulate(reverse):
        def signals(_, ts):
            if ts == 0:
                return [{'ticker': 'OLD', 'signal': 'BUY'}]
            if ts == 2:
                # The entry must have access to proceeds from OLD's exit.
                result = [{'ticker': 'NEW', 'signal': 'BUY'},
                          {'ticker': 'OLD', 'signal': 'SELL'}]
                return list(reversed(result)) if reverse else result
            return []
        with patch('backtest.momentum_signals', side_effect=signals):
            return run_backtest(prices, start_index=0)

    first, second = simulate(False), simulate(True)
    assert first['equity_curve'].equals(second['equity_curve'])
    assert first['trades'][-1]['ticker'] == 'NEW'
    assert first['trades'][-1]['shares'] == 7500
    assert first['equity_curve'].iloc[-1] == 165000.
