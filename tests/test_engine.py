import pytest
import pandas as pd
import numpy as np
import os
from engine.data import get_data
from engine.utils import compute_log_returns, normalize_signal

def test_data_fetching():
    # Test with a known ticker already cached
    df = get_data("SPY")
    assert df is not None
    assert not df.empty
    assert "Close" in df.columns

def test_log_returns():
    data = pd.Series([100, 110, 121])
    returns = compute_log_returns(data)
    # log(1.1) approx 0.0953
    # log(1.21/1.1) = log(1.1)
    assert len(returns) == 3
    assert returns[0] == 0
    assert np.isclose(returns[1], np.log(1.1))

def test_normalization():
    signal = np.array([1, 2, 3, 4, 5])
    norm = normalize_signal(signal)
    assert np.isclose(np.mean(norm), 0)
    assert np.isclose(np.std(norm), 1)

if __name__ == "__main__":
    pytest.main([__file__])
