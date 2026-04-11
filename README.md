# Market DNA Engine 🧬
### Institutional-Grade Signal Processing for Alpha Generation

Standard quantitative models are failing because they oversimplify the market into a single time-series. The **Market DNA Engine** utilizes Digital Signal Processing (DSP) to mathematically dissect market volatility into its component frequencies—separating the fast, algorithmic "Noise" from the deep, structural "Macro Trends."

---

## 🔬 The Core Thesis: Signal vs. Noise
Linear indicators like Moving Averages (MA) and RSI are fundamentally flawed because they inherit **lag**. By the time an MA-cross happens, the structural shift is already over.

The Market DNA Engine utilizes **Orthogonal Wavelet Decomposition** and **Synchrosqueezing** to isolate market cycles *without phase shift*. This allows for the identification of structural support and resistance before they manifest in lagging technical indicators.

---

## 🛠️ Integrated Modules

### 1. Price DNA Explorer
Break down price action into fundamental structural bands:
- **Structural Trend (Macro)**: The deep, underlying economic direction.
- **Cycle Momentum (Weekly)**: Institutional swing waves.
- **Micro-Noise (Intraday)**: High-frequency algorithmic volatility.
- **Includes**: *Math Integrity Proof* (99.9% reconstruction accuracy).

### 2. Spectral Resonance Guardian
Traditional correlation measures total movement. **Resonance** measures exactly *which frequencies* assets share. Identify hidden systemic risks when multiple unrelated assets start "vibrating" on the same macro frequency.

### 3. Causal Leadership Flow
Stop guessing who leads the market. Use **Spectral Granger Causality** to uncover hidden directional leadership between assets at specific time scales.

---

## 🚀 Quick Launch

### Local Deployment (3 Minutes)
1. **Environment Setup**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Pre-load Intelligence Cache**:
   ```powershell
   python cache_data.py
   ```

3. **Launch the Engine**:
   ```powershell
   streamlit run 0_Home.py
   ```

### 📡 Production API Integration
The engine is decoupled via **FastAPI**. Hook up your HFT bots to pull real-time spectral metrics:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🏗️ The Tech Stack
- **Math Architecture**: PyWavelets (DWT), ssqueezepy (Synchrosqueezing), pycwt (Coherence).
- **Inference Engine**: Statsmodels (VAR-based Causality).
- **UI/UX**: Custom Glassmorphism Streamlit Design System.
- **Backend**: FastAPI / Pydantic.

---
### 🛰️ AlgoFest 2026 Submission
*Built by Josh Chomiak. Securing Alpha through Math Integrity.*
