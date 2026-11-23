import pandas as pd
import pytest

from backtest import run_backtest
from momentum_strategy import momentum_signals


@pytest.mark.parametrize("length", [3, 70])
@pytest.mark.parametrize("entrypoint", ["backtest", "signals"])
def test_duplicate_tickers_are_rejected_before_warmup_or_simulation(length, entrypoint):
    prices = pd.DataFrame([[100., 100., 200.]] * length,
                          index=pd.date_range("2025-01-01", periods=length),
                          columns=["AAA", "AAA", "BBB"])
    with pytest.raises(ValueError, match="Ticker columns must be unique"):
        if entrypoint == "backtest":
            run_backtest(prices, start_index=0)
        else:
            momentum_signals(prices, prices.index[-1])


def test_unique_ticker_benchmark_keeps_equal_weighting():
    prices = pd.DataFrame({"AAA": [100., 110., 120.], "BBB": [200., 200., 200.]},
                          index=pd.date_range("2025-01-01", periods=3))
    result = run_backtest(prices, start_index=0)
    assert result["benchmark_curve"].tolist() == pytest.approx([100000., 105000., 110000.])
