import pandas as pd
import pytest
from backtest import run_backtest
from momentum_strategy import momentum_signals

@pytest.mark.parametrize('index', [[2,0,1], [0,1,1]])
@pytest.mark.parametrize('function', [lambda p: run_backtest(p, start_index=0), lambda p: momentum_signals(p, 1)])
def test_ambiguous_chronology_is_rejected(index, function):
    with pytest.raises(ValueError, match='unique and increasing'):
        function(pd.DataFrame({'A':[100.,101.,102.]}, index=index))

def test_signal_timestamp_must_be_an_observed_bar():
    with pytest.raises(ValueError, match='timestamp'):
        momentum_signals(pd.DataFrame({'A':[100.,101.]}, index=[0,2]), 1)
