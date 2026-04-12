---
title: Market DNA Engine
emoji: 📊
colorFrom: gray
colorTo: green
sdk: streamlit
app_file: 0_Home.py
pinned: false
license: mit
---

# Market DNA Engine
### Institutional Digital Signal Processing for Technical Alpha

The Market DNA Engine utilizes Digital Signal Processing (DSP) to decompose market volatility into specific frequency components. This separates algorithmic noise from structural macro trends.

---

## Core Methodology: Signal Extraction
Linear indicators inherit phase lag. The Market DNA Engine utilizes Orthogonal Wavelet Decomposition and Synchrosqueezing to isolate market cycles without phase shift. This allows for the identification of structural support and resistance prior to manifestation in lagging technical indicators.

---

### Core Analytical Modules

1. **Spectral Decomposition**: Orthogonal signal extraction via Multiresolution Analysis (MRA).
2. **Cross-Spectral Coherence**: Statistical resonance tracking for identifying leading market indicators.
3. **Directional Causality**: Evaluation of spectral precedence via lead-lag phase estimation.
4. **Institutional Backtesting**: Performance simulation with lead-adjusted execution, Kelly Criterion allocation, and risk metrics (Sharpe, Sortino, Calmar, Profit Factor).
5. **Resonance Guardian**: Real-time portfolio monitoring to detect frequency-domain instability.

---

## Integrated Modules

### 1. Price Decomposition
Extraction of price action into fundamental structural bands:
- **Structural Trend (Macro)**: Underlying economic direction.
- **Cycle Momentum (Intermediate)**: Swing waves.
- **Micro-Volatility (Intraday)**: High-frequency algorithmic noise.
- **Validation**: Includes Math Integrity Proof (99.9%+ reconstruction accuracy).

### 2. Spectral Resonance
Measures frequency-domain cross-correlation. Identify systemic risk when distinct assets exhibit shared macro-frequency resonance.

### 3. Spectral Causality
Directional leadership detection using Spectral Granger Causality. Uncover hidden leadership between assets at specific time scales.

---

## Deployment

### Local Environment
1. **Setup**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Initialize Cache**:
   ```powershell
   python cache_data.py
   ```

3. **Execution**:
   ```powershell
   streamlit run 0_Home.py
   ```

### API Integration
The engine is accessible via FastAPI for external integration.
**[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## Architecture
- **Processing**: PyWavelets (DWT), ssqueezepy (Synchrosqueezing), pycwt (Coherence).
- **Inference**: Statsmodels (VAR-based Causality).
- **Interface**: Streamlit Institutional Slate Theme.
- **Backend**: FastAPI.

---
### AlgoFest 2026 Submission
*Market DNA Engine: Alpha through Signal Processing.*
