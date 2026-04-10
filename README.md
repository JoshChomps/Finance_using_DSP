# FinSignal Suite — Multi-Resolution Financial Signal Processing Engine

FinSignal Suite is a real-time multi-resolution spectral analysis engine that decomposes financial asset volatility into frequency bands, detects cross-asset resonance, and reveals directional causal coupling.

## Project Structure

```
├── app.py                          # Streamlit entry point
├── pages/                          # Dashboard panels
├── engine/                         # Core algorithmic engine
│   ├── data.py                     # Data ingestion and caching
│   └── utils.py                    # Signal processing helpers
├── data/
│   └── cache/                      # Pre-computed market data
├── tests/                          # Unit and integration tests
├── docs/                           # Documentation
└── requirements.txt                # Project dependencies
```

## Getting Started

1. **Set up environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Pre-cache data (optional but recommended):**
   ```powershell
   python cache_data.py
   ```

3. **Run the dashboard:**
   ```powershell
   streamlit run app.py
   ```

## Core Methodology

Built on signal processing fundamentals:
- **Wavelet Decomposition (DWT):** Separates price series into macro, weekly, and noise bands.
- **Wavelet Coherence (WTC):** Identifies frequency-localized coupling between assets.
- **Spectral Granger Causality:** Detects leading/lagging relationships across frequency bands.
