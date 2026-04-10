import streamlit as st

st.set_page_config(
    page_title="FinSignal Suite",
    page_icon="📈",
    layout="wide",
)

st.title("🛡️ FinSignal Suite")
st.subheader("Multi-Resolution Financial Signal Processing Engine")

st.markdown("""
### Welcome to the Future of Market Risk Analysis
Traditional correlation metrics treat all timescales identically, creating dangerous blind spots. 
**FinSignal Suite** uses advanced Digital Signal Processing (DSP) to decompose market volatility 
into distinct frequency bands, revealing hidden structures that standard metrics miss.

#### Core Modules:
1. **🔍 Decomposition Explorer**: Break down price action into Macro, Weekly, and Noise components using Wavelets.
2. **🤝 Cross-Asset Resonance**: Identify frequency-localized coupling between assets—exactly when and where decoupling fails.
3. **➡️ Directional Causality**: Reveal which asset leads and which follows using Spectral Granger Causality.

---
### Getting Started
Use the sidebar to navigate between the different analysis modules. Select tickers, adjust wavelet parameters, 
and explore the hidden frequency structure of the markets.

*Powered by PyWavelets, ssqueezepy, and Geweke Spectral Analysis.*
""")

# Sidebar info
st.sidebar.info("Select a module above to begin your analysis.")
st.sidebar.markdown("---")
st.sidebar.caption("FinSignal Suite v1.0 | AlgoFest 2026")
