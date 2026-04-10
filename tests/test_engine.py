import pytest
import pandas as pd
import numpy as np
import os
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize

def test_data_fetching():
    # Test with a known symbol already cached
    prices = get_data("SPY")
    assert prices is not None
    assert not prices.empty
    assert "Close" in prices.columns

def test_calculate_returns():
    history = pd.Series([100, 110, 121])
    returns = calculate_returns(history)
    # log(1.1) approx 0.0953
    assert len(returns) == 3
    assert returns[0] == 0
    assert np.isclose(returns[1], np.log(1.1))

def test_z_score_normalization():
    raw_signal = np.array([1, 2, 3, 4, 5])
    standardized = z_score_normalize(raw_signal)
    assert np.isclose(np.mean(standardized), 0)
    assert np.isclose(np.std(standardized), 1)

if __name__ == "__main__":
    pytest.main([__file__])
