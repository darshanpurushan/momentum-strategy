from unittest.mock import patch
import pandas as pd
import pytest
import tune

def result(*args, **kwargs):
    return {'metrics':{'strategy':dict(total_return=0.,cagr=0.,sharpe=.5,max_drawdown=0.)},'trades':[]}

def test_candidates_use_identical_evaluation_dates():
    with patch.multiple(tune, SHORT_LONG_COMBOS=[(1,2),(2,4)], QUALITY_SCORE_THRESHOLDS=[.5], MIN_PCT_POSITIVE_DAYS_OPTIONS=[.5]), patch.object(tune,'run_backtest',side_effect=result) as run:
        tune.sweep(pd.DataFrame({'A':range(20)}), quality_window_days=3)
        assert [call.kwargs['start_index'] for call in run.call_args_list] == [5,5]

def test_too_short_common_window_is_rejected_before_search():
    with patch.multiple(tune, SHORT_LONG_COMBOS=[(1,2),(2,4)]), patch.object(tune,'run_backtest',side_effect=result) as run:
        with pytest.raises(ValueError, match='common evaluation'):
            tune.sweep(pd.DataFrame({'A':range(4)}), quality_window_days=3)
        run.assert_not_called()
