# Market DNA Engine
### Institutional Digital Signal Processing for Non-Lagging Alpha Extraction

The Market DNA Engine is a professional-grade strategy analysis platform that utilizes Digital Signal Processing (DSP) to decompose market price action into orthogonal frequency components. This architecture separates structural macro trends from stochastic algorithmic noise, providing a non-lagging foundation for institutional decision-making.

---

## Live Demo

### 1. Problem Identification
Traditional technical indicators (Moving Averages, RSI, MACD) are fundamentally reactive. They rely on lookback windows that introduce **phase lag**-the delay between a market event and the indicator's signal. In modern high-frequency regimes, this delay leads to "whipsaw" entries and delayed exits, eroding alpha. Furthermore, raw price data is "noisy," combining long-term value shifts with short-term liquidity distortions.

### 2. Innovative Approach
Market DNA solves the lag problem by shifting analysis from the Time Domain to the **Frequency Domain**. By using **Orthogonal Wavelet Decomposition (MRA)** and **Synchrosqueezing**, the engine extracts specific "Market Rhythms" (cycles) without the temporal distortion inherent in traditional filters. This allows traders to identify structural support and cyclical exhaustion *before* they manifest in lagging indicators.

### 3. Technical Implementation
- **Signal Extraction**: Multiresolution Analysis (MRA) using Daubechies (db4) and Symlets (sym8) wavelets.
- **Resonance Tracking**: Cross-Wavelet Coherence (CWT) for tracking leading/lagging relationships between assets (e.g., BTC as a leading indicator for SPY).
- **Causal Inference**: Vector Autoregression (VAR) based Spectral Granger Causality to identify directional information flow.
- **Risk Management**: Integrated Backtesting suite with Kelly Criterion capital allocation and OOS (Out-of-Sample) validation.

### 4. Usability and User Experience
The platform features a high-fidelity **Slate-Carbon Tactical HUD**, designed for high-density information display. It provides clinical "Execution Playbooks" that translate complex DSP waveforms into actionable 1-2-3 trading instructions.

### 5. Scalability and Feasibility
The engine is built on a modular data-agnostic layer. It defaults to public Yahoo Finance data for accessibility but is fully bridged for institutional APIs (Alpaca, Polygon, Interactive Brokers). The DSP logic is optimized via NumPy/SciPy for sub-second analysis on standard commodity hardware.

---

## Technologies Used
- **Language**: Python 3.10+
- **Frontend**: Streamlit (Framework), Vanilla CSS (Custom HUD)
- **DSP Core**: PyWavelets, ssqueezepy, pycwt
- **Analysis**: Statsmodels, Pandas, NumPy
- **Visualization**: Plotly (Interactive Spectral Charts)
- **Deployment**: Hugging Face Spaces, GitHub Actions

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JoshChomps/Finance_using_DSP.git
   cd Finance_using_DSP
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Locally**:
   ```bash
   streamlit run 0_Home.py
   ```

---

## Team Details
- **Josh Chomiak**: Lead Engineer & Quantitative Architect
  - Contribution: End-to-end development of the DSP engine, HUD interface, and backtesting framework.

---
**Institutional Standard | Algofest 2026 | Systematic Alpha.**
