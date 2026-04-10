from engine.data import pre_cache_tickers
import os

# Essential tickers according to plan.md
tickers = [
    'SPY', 'QQQ', 'GLD', 'TLT',  # Asset classes
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META',  # Big Tech
    'XLE', 'XLF', 'XLK', 'XLV',  # Sectors
    'BTC-USD', 'ETH-USD', # Crypto
    'JPM', 'V', 'PG', 'UNH' # Blue chip
]

if __name__ == "__main__":
    print(f"Starting pre-caching for {len(tickers)} tickers...")
    results = pre_cache_tickers(tickers)
    for ticker, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{ticker}: {status}")
