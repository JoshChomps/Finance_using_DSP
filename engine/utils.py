import numpy as np
import pandas as pd

def calculate_returns(prices, column='Close'):
    """
    Turns price data into log returns. This makes the data easier to 
    process for signal analysis since it stabilizes the variance.
    """
    if isinstance(prices, pd.DataFrame):
        target_series = prices[column]
    else:
        target_series = prices
    
    returns = np.log(target_series / target_series.shift(1))
    return returns.fillna(0)

def z_score_normalize(series):
    """
    Centers the series at zero and scales it so the standard deviation is 1.
    Very useful for comparing two different stocks that trade at different prices.
    """
    return (series - np.mean(series)) / np.std(series)

def apply_taper(series, window='hann'):
    """
    Applies a taper (window) to the edges of the signal to prevent leakage 
    when we perform spectral transforms.
    """
    from scipy.signal import get_window
    taper_weights = get_window(window, len(series))
    return series * taper_weights
