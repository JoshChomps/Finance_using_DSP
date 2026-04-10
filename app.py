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

st.title("🛡️ FinSignal Suite")
st.subheader("Signal Processing for the Financial Markets")

st.markdown("""
### Welcome to the Future of Market Risk Analysis
Most people look at correlation as a single number, but that creates dangerous blind spots. 
**FinSignal Suite** uses Digital Signal Processing (DSP) to break market volatility 
into distinct cycles, revealing the hidden structures that standard metrics miss.

#### Core Modules:
1. **🔍 Decomposition Explorer**: Break down price action into Macro, Weekly, and Noise components using Wavelets.
2. **🤝 Cross-Asset Resonance**: Identify exactly when and where two assets are moving in sync at specific frequencies.
3. **➡️ Directional Causality**: Find out which asset is leading and which is following using Spectral Granger analysis.

---
### Getting Started
Use the sidebar to jump between modules. Pick your symbols, adjust the analysis depth, and start exploring the hidden underlying cycles of the market.

*Built on PyWavelets, ssqueezepy, and Geweke Spectral Analysis.*
""")

# Sidebar info
st.sidebar.info("Select a module above to begin your analysis.")
st.sidebar.markdown("---")
st.sidebar.caption("FinSignal Suite v1.0 | AlgoFest 2026")
