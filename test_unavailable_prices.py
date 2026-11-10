from unittest.mock import patch
import pandas as pd
import pytest
from data_loader import load_prices
import validate

@pytest.mark.parametrize('tickers', ['AAPL',['AAPL','MSFT']])
def test_empty_vendor_response_preserves_empty_price_schema(tickers):
    with patch('data_loader.yf.download', return_value=pd.DataFrame()):
        out = load_prices(tickers, period='1y')
    assert out.empty
    assert out.columns.tolist() == ([tickers] if isinstance(tickers,str) else tickers)

def test_all_missing_requested_tickers_returns_empty_prices():
    raw = pd.DataFrame([[1.]], columns=pd.MultiIndex.from_tuples([('OTHER','Close')]))
    with patch('data_loader.yf.download', return_value=raw):
        assert load_prices(['AAPL','MSFT']).empty

def test_all_skipped_validation_periods_return_an_empty_report():
    with patch.object(validate,'load_prices',return_value=pd.DataFrame()):
        out = validate.validate()
    assert out.empty and out.index.name == 'period'
    assert 'strategy_return' in out.columns
