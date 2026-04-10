import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from functools import lru_cache

# Where we store our local market data downloads
# Allow override via environment variable for production/Docker environments
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')
CACHE_FOLDER = os.getenv('FIN_DATA_CACHE', DEFAULT_CACHE)

@lru_cache(maxsize=32)
def get_data(symbol, period='5y', interval='1d', use_cache=True):
    """
    Grabs historical price data. We check the local cache first to save time 
    and avoid hitting rate limits.
    """
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)

    file_path = os.path.join(CACHE_FOLDER, f"{symbol}_{period}_{interval}.parquet")

    if use_cache and os.path.exists(file_path):
        print(f"Loading {symbol} from local storage...")
        cached = pd.read_parquet(file_path)
        if isinstance(cached.columns, pd.MultiIndex):
            cached.columns = cached.columns.get_level_values(0)
        return cached

    print(f"Downloading {symbol} from yfinance...")
    try:
        prices = yf.download(symbol, period=period, interval=interval, progress=False)
        if prices.empty:
            raise ValueError(f"Could not find any data for {symbol}")

        # Newer yfinance (>=0.2.38) returns MultiIndex columns like ('Close', 'SPY').
        # Flatten to simple column names so downstream code works with both versions.
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
            
        # Deduplicate columns if they exist (sometimes yfinance returns redundant stacks)
        prices = prices.loc[:, ~prices.columns.duplicated()].copy()

        # Save a local copy for next time
        prices.to_parquet(file_path)
        return prices
    except Exception as e:
        print(f"Oops, ran into an issue fetching {symbol}: {e}")
        
        # Last ditch effort: try to find any existing file for this asset
        for filename in os.listdir(CACHE_FOLDER):
            if filename.startswith(symbol) and filename.endswith(".parquet"):
                print(f"Using old cache as backup: {filename}")
                fallback = pd.read_parquet(os.path.join(CACHE_FOLDER, filename))
                if isinstance(fallback.columns, pd.MultiIndex):
                    fallback.columns = fallback.columns.get_level_values(0)
                return fallback
        return None

def pre_cache_list(symbols, period='5y', interval='1d'):
    """
    Helper to bulk download a list of assets.
    """
    status = {}
    for sym in symbols:
        data = get_data(sym, period=period, interval=interval)
        status[sym] = data is not None
    return status
