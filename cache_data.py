"""
cache_data.py - Pre-downloads market data for every symbol used in the dashboard.

Run this once before launching the app so every page loads instantly:
    python cache_data.py

Data is saved as Parquet files in data/cache/ and reused until you delete them.
Re-running this script refreshes all data from Yahoo Finance.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from engine.data import get_data

# All symbols referenced anywhere in the dashboard or API
ASSETS = [
    # Major asset classes
    "SPY", "QQQ", "GLD", "TLT",
    # Big Tech
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META",
    # Sector ETFs
    "XLE", "XLF", "XLK", "XLV",
    # Crypto
    "BTC-USD", "ETH-USD",
    # Institutional blue chips
    "JPM", "V", "PG", "UNH",
]


def _fetch(symbol):
    t0 = time.time()
    data = get_data(symbol, use_cache=False)  # force a fresh download
    elapsed = time.time() - t0
    return symbol, data is not None, elapsed


def main():
    total = len(ASSETS)
    print(f"\nPre-caching {total} assets into local storage (parallel download)...\n")

    results = {}
    start = time.time()

    for i, sym in enumerate(ASSETS, 1):
        sym, ok, elapsed = _fetch(sym)
        status = "OK" if ok else "FAILED"
        print(f"  [{i:2d}/{total}] {sym:<10} {status}  ({elapsed:.1f}s)")
        results[sym] = ok

    wall = time.time() - start
    passed = sum(results.values())
    failed = [s for s, ok in results.items() if not ok]

    print(f"\n{'-' * 40}")
    print(f"Done in {wall:.1f}s - {passed}/{total} symbols cached.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print("These symbols may be delisted or temporarily unavailable.")
    print()


if __name__ == "__main__":
    main()
