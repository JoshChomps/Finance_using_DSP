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

## 📡 Hooking up the API

FinSignal Suite perfectly decouples the underlying DSP math engine from the visual UI via **FastAPI** so you can seamlessly pipe analysis directly into Python scripts, Node.js servers, or algorithmic trading bots.

### 1. The Interactive Docs (Swagger)
The clearest way to see what the API expects and returns is to use the auto-generated Swagger UI. While the backend (`uvicorn`) is running, open your browser to:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

From there, you can view the required JSON schemas and execute live requests directly from your browser.

### 2. Python Integration Example
If you want to trigger the remote DSP engine before your bot executes a routine trade (e.g., checking the Spectral Granger Causality between two assets to confirm macro leadership), you just send a standard HTTP `POST` request.

```python
import requests

# Point to your FinSignal Engine's IP / Port
url = "http://localhost:8000/api/v1/causality"

# Define the assets you want to analyze
payload = {
    "first": "AAPL",
    "second": "QQQ"
}

# 1. Ping the causal engine
response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    
    # 2. Extract the max causal strength block
    fwd_flow = data["causal_strength_fwd"]
    
    print(f"Algorithm Ready:")
    print(f"{data['candidate']} -> {data['target']} max flow strength is {max(fwd_flow):.3f}")
else:
    print("API Error:", response.text)
```

**Currently Available Endpoints:**
- `POST /api/v1/decompose` (Body: `{ "symbol": "SPY" }`)
- `POST /api/v1/coherence` (Body: `{ "first": "SPY", "second": "TLT" }`)
- `POST /api/v1/causality` (Body: `{ "first": "SPY", "second": "TLT" }`)


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
