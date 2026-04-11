import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Market DNA Engine",
    layout="wide",
)

from engine.ui import inject_custom_css
inject_custom_css(st)

st.title("Market DNA Engine")
st.subheader("Orthogonal Frequency Analysis for Institutional Alpha")

st.markdown("""
<br>

### Signal Extraction and Noise Mitigation
Standard quantitative models often fail due to the aggregation of disparate market cycles into a single time-series. In practice, market price is a composite signal: a combination of high-frequency algorithmic noise, intermediate swing momentum, and long-term structural macro cycles. 

The Market DNA Engine utilizes Digital Signal Processing (DSP) to isolate these components. Through Wavelet Decomposition and Synchrosqueezing, the engine extracts the underlying cycles that drive structural trends.

---
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Price Analysis")
    st.write("Decomposition of price action into fundamental components. Isolate structural macro trends from intraday high-frequency volatility.")

with col2:
    st.markdown("#### Spectral Resonance")
    st.write("Identification of shared frequencies between assets. Measures systemic risk through frequency-domain cross-correlation.")

with col3:
    st.markdown("#### Causal Flow")
    st.write("Determination of directional leadership through Spectral Granger Causality. Identify leading indicators across specific time scales.")

st.markdown("""
---
### Strategic Indicators
By isolating the structural trend from market noise, the engine provides the following capabilities:
- **Volatility Filtering**: Distinction between price movement driven by high-frequency noise and movement supported by macro cycles.
- **Resonance Management**: Detection of portfolio over-exposure to specific frequency bands.
- **Leadership Detection**: Identification of leadership shifts prior to manifestation in lagging technical indicators.

---
### Technical Specifications: AlgoFest 2026
Navigation is available through the sidebar. Each module includes a Math Integrity Validation and a Stance Decoder.
""", unsafe_allow_html=True)

st.sidebar.markdown('**Market DNA v1.2**')
st.sidebar.success("Technical Execution Verified")
st.sidebar.info("Select a module above.")
st.sidebar.caption("Institutional Alpha through Digital Signal Processing.")
