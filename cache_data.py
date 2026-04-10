from engine.data import pre_cache_list
import os

# Assets we want to have ready for the dashboard
assets = [
    'SPY', 'QQQ', 'GLD', 'TLT',  # Major classes
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META',  # Big Tech
    'XLE', 'XLF', 'XLK', 'XLV',  # Sector ETFs
    'BTC-USD', 'ETH-USD', # Crypto basics
    'JPM', 'V', 'PG', 'UNH' # Institutional blue chips
]

if __name__ == "__main__":
    print(f"Pre-loading {len(assets)} assets into local storage...")
    status_map = pre_cache_list(assets)
    for sym, success in status_map.items():
        outcome = "Done" if success else "Failed"
        print(f"{sym}: {outcome}")
