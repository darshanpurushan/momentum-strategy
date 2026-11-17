import numpy as np
import pandas as pd
import pytest

from momentum_quality import classify_momentum_quality, rolling_momentum_quality


@pytest.mark.parametrize('calculate', [classify_momentum_quality, rolling_momentum_quality])
@pytest.mark.parametrize('invalid', [0., -1., np.inf, -np.inf])
def test_quality_rejects_prices_outside_return_domain(calculate, invalid):
    prices = pd.DataFrame({'A': [100., 102., invalid, 103., 104.]})
    with pytest.raises(ValueError, match='finite positive'):
        calculate(prices, 'A', windows={'test': 3})


def test_missing_quotes_remain_supported_as_undefined_quality():
    prices = pd.DataFrame({'A': [100., 102., np.nan, 103., 104.]})
    snapshot = classify_momentum_quality(prices, 'A', windows={'test': 3})
    rolling = rolling_momentum_quality(prices, 'A', windows={'test': 3})
    assert pd.isna(snapshot.loc['test', 'quality'])
    assert pd.isna(rolling['test'].iloc[-1]['quality'])
