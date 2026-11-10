from unittest.mock import patch
import pandas as pd
import tune

def test_undefined_sharpe_results_rank_after_finite_results():
    results=[]
    for score in [float('nan'), .5, None]:
        results.append({'metrics':{'strategy':dict(total_return=0.,cagr=0.,sharpe=score,max_drawdown=0.)},'trades':[]})
    with patch.multiple(tune, SHORT_LONG_COMBOS=[(1,2),(2,3),(3,4)], QUALITY_SCORE_THRESHOLDS=[.5], MIN_PCT_POSITIVE_DAYS_OPTIONS=[.5]), patch.object(tune,'run_backtest',side_effect=results):
        out=tune.sweep(pd.DataFrame({'A':range(20)}), quality_window_days=3)
    assert out[0]['sharpe'] == .5

def test_undefined_scores_can_be_printed_without_numeric_format_errors():
    for score in (None,float('nan'),float('inf')):
        assert tune.display_sharpe(score) == 'n/a'
    assert tune.display_sharpe(.25) == '0.2500'
