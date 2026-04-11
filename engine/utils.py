import numpy as np
import pandas as pd

def calculate_returns(prices, column='Close'):
    """
    Turns price data into log returns. This makes the data easier to
    process for signal analysis since it stabilizes the variance.
    """
    if isinstance(prices, pd.DataFrame):
        col_data = prices[column]
        # Guard against duplicate columns returning a DataFrame instead of a Series
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        target_series = col_data
    else:
        target_series = prices

    returns = np.log(target_series / target_series.shift(1))
    return returns.fillna(0)

def z_score_normalize(series):
    """
    Centers the series at zero and scales it so the standard deviation is 1.
    Very useful for comparing two different stocks that trade at different prices.
    Returns a zero-centered series unchanged when std is effectively zero.
    """
    mu = np.mean(series)
    sigma = np.std(series)
    if sigma < 1e-10:
        return series - mu
    return (series - mu) / sigma

def apply_taper(series, window='hann'):
    """
    Applies a taper (window) to the edges of the signal to prevent leakage 
    when we perform spectral transforms.
    """
    from scipy.signal import get_window
    taper_weights = get_window(window, len(series))
    return series * taper_weights
