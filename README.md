# FinSignal Suite
### Signal Processing for the Modern Market

Most financial models look at the market through a single lens. FinSignal Suite uses Digital Signal Processing (DSP) to peel back the layers of price action, revealing the hidden cycles and "resonance" between assets that standard tools miss.

## Quick Setup

Getting the engine running on your local machine takes about 3 minutes.

1. **Environment Setup**:
   Create a virtual environment and install the core stack.
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Pre-load Market Data**:
   We've included a script to download and cache historical data for the S&P 500, Gold, and big tech.
   ```powershell
   python cache_data.py
   ```

3. **Launch the Dashboard**:
   ```powershell
   streamlit run 0_Home.py
   ```

4. **Launch the REST API**:
   If you want to pull these insights into an external bot (like an HFT system):
   ```powershell
   uvicorn api.main:app --reload
   ```

## Testing the Engine
We use `pytest` to make sure the signal decomposition and causal logic are working perfectly.
```powershell
pytest tests/
```

## The Data Strategy (Where to find it)
The biggest hurdle in quant research is good data. We currently use `yfinance` because it is free and easy, but here is how you should graduate for production testing:

*   **For High-Frequency (1m data)**: Use **Polygon.io**. It is the gold standard for retail devs. If you're on a budget, **Alpaca** offers decent historical data on their free tier.
*   **For Long-term Macro (20+ years)**: Check **Kaggle**. There are massive CSV dumps of the entire history of symbols like SPY and QQQ.
*   **For Crypto**: Most exchanges like **Binance** or **Coinbase** offer public archives of every single tick for years. These are perfect for training the "Resonance" model.

## The Tech Stack
- **Python 3.10+**
- **FastAPI**: The backbone for our REST endpoints.
- **Streamlit**: The visual command center.
- **PyWavelets**: The core math for orthogonal decomposition.
- **ssqueezepy**: For high-resolution synchrosqueezing.
- **pycwt**: For spectral cross-metrics (coherence).
- **statsmodels**: For the VAR / Granger causality logic.
