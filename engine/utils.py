import numpy as np
import pandas as pd

def compute_log_returns(data, column='Close'):
    """
    Compute log returns for a given price series.
    Returns a Series with the first element as 0 or NaN.
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.fillna(0)

def normalize_signal(signal):
    """
    Normalize a signal to zero mean and unit variance.
    """
    return (signal - np.mean(signal)) / np.std(signal)

def apply_windowing(signal, window_type='hann'):
    """
    Apply a window function to a signal.
    """
    from scipy.signal import get_window
    window = get_window(window_type, len(signal))
    return signal * window
