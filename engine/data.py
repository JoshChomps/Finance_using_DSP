import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')

def get_data(ticker, period='5y', interval='1d', use_cache=True):
    """
    Fetch historical data for a given ticker.
    Uses local cache if available and requested.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    file_path = os.path.join(DATA_DIR, f"{ticker}_{period}_{interval}.parquet")

    if use_cache and os.path.exists(file_path):
        print(f"Loading {ticker} from cache...")
        return pd.read_parquet(file_path)

    print(f"Fetching {ticker} from yfinance...")
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Ensure data is saved in a consistent format
        data.to_parquet(file_path)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        # Try to find any existing cache for this ticker as fallback
        for file in os.listdir(DATA_DIR):
            if file.startswith(ticker) and file.endswith(".parquet"):
                print(f"Fallback: Loading alternative cache for {ticker} from {file}")
                return pd.read_parquet(os.path.join(DATA_DIR, file))
        return None

def pre_cache_tickers(tickers, period='5y', interval='1d'):
    """
    Download and cache data for a list of tickers.
    """
    results = {}
    for ticker in tickers:
        data = get_data(ticker, period=period, interval=interval)
        results[ticker] = data is not None
    return results
