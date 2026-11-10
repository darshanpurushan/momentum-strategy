import pandas as pd
from momentum_quality import classify_momentum_quality, rolling_momentum_quality

def test_missing_bars_are_not_compressed_into_a_complete_window():
    prices = pd.DataFrame({'A':[100.,101.,float('nan'),103.,104.,105.]})
    out = classify_momentum_quality(prices,'A',windows={'long':3,'short':2})
    assert pd.isna(out.loc['long','quality'])
    assert pd.notna(out.loc['short','quality'])
    assert out.loc['short','momentum_return'] == 105./103.-1
    assert pd.isna(rolling_momentum_quality(prices,'A',windows={'long':3})['long'].iloc[-1]['quality'])

def test_missing_latest_bar_does_not_report_an_old_quality():
    prices = pd.DataFrame({'A':[100.,101.,103.,104.,float('nan')]})
    assert pd.isna(classify_momentum_quality(prices,'A',windows={'test':3}).loc['test','quality'])
