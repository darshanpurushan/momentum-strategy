from unittest.mock import patch
import pandas as pd
import pytest
import momentum_strategy as strategy

@pytest.mark.parametrize("values", [[5., 4., 3., float('nan')], [1., 2., 3., float('nan')]])
def test_missing_current_quote_cannot_emit_a_stale_signal(values):
    prices = pd.DataFrame({'A': values}, index=pd.date_range('2025-01-01', periods=4))
    with patch.multiple(strategy, SHORT_WINDOW=1, LONG_WINDOW=2, QUALITY_WINDOW_DAYS=2), patch.object(strategy, 'classify_momentum_quality', return_value=pd.DataFrame({'quality':['HIGH']}, index=['3m'])):
        assert strategy.momentum_signals(prices, prices.index[-1]) == []

def test_valid_current_quote_still_generates_an_exit():
    prices = pd.DataFrame({'A':[4.,3.,2.]})
    with patch.multiple(strategy, SHORT_WINDOW=1, LONG_WINDOW=2):
        assert strategy.momentum_signals(prices, 2)[0]['signal'] == 'SELL'
