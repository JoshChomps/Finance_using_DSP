import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="FinSignal Suite",
    page_icon="📈",
    layout="wide",
)

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("FinSignal Suite")
st.subheader("Signal Processing for the Financial Markets")

st.markdown("""
### Welcome to the Future of Market Risk Analysis
Standard metrics like moving averages or linear correlations look at the market through an oversimplified lens, leading to dangerous blind spots during structural changes. 

**FinSignal Suite** utilizes Digital Signal Processing (DSP) to mathematically dissect market volatility into its component frequencies. Just like a prism splits white light into a rainbow, our tool utilizes Wavelets and Spectral Causality to break down complex price movements into isolated microscopic and macroscopic cycles.

#### 🧠 Why DSP in Finance?
- **Phase Alignment**: Moving averages inherently lag. Wavelets can isolate dominant trends *without* shifting them back in time, allowing you to see support and resistance before they manifest in lagging indicators.
- **Granular Correlation**: Traditional correlation checks if two assets move together overall. DSP **Resonance** checks exactly *which specific types of movements* (e.g., short-term panics vs multi-month trends) they share.

#### Core Modules:
1. **Decomposition Explorer**: Break down price action into Macro, Weekly, and Noise components using localized Wavelet filtering. See the "true trend" without the noise.
2. **Cross-Asset Resonance**: Identify exactly when and where two assets are moving in sync at specific frequencies.
3. **Directional Causality**: Uncover hidden leadership. Find out which asset is leading the market by measuring Spectral Granger Causality.

---
### 🚀 Getting Started
Use the sidebar to jump between modules. Pick your symbols, adjust the analysis depth, and start exploring the hidden underlying cycles of the market.

*Built on PyWavelets, ssqueezepy, and Geweke Spectral Analysis.*
""")

# Sidebar info
st.sidebar.info("Select a module above to begin your analysis.")
st.sidebar.markdown("---")
st.sidebar.caption("FinSignal Suite v1.0 | AlgoFest 2026")
