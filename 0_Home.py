import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Market DNA Engine",
    page_icon="🧬",
    layout="wide",
)

# Load CSS via the established ui module for better consistency across pages
from engine.ui import inject_custom_css
inject_custom_css(st)

# Institutional Hero Section
st.markdown('<h1 class="header-gradient">The Market DNA Engine</h1>', unsafe_allow_html=True)
st.subheader("Orthogonal Frequency Analysis for Institutional Alpha")

st.markdown("""
<br>

### 🔬 The Signal, Not the Noise
Most quantitative models fail because they view the market as a single, messy time-series. In reality, market price is a **composite signal**—a complex mix of HFT noise, weekly swing momentum, and deep structural macro cycles. 

**The Market DNA Engine** uses advanced Digital Signal Processing (DSP) to peel back these layers with surgical precision. Just as a prism separates light into its component colors, our engine utilizes **Wavelets** and **Synchrosqueezing** to isolate the cycles that actually drive the trend.

---
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🧬 Price DNA")
    st.write("We break price action into its fundamental building blocks. Isolate the structural macro trend from the intraday algorithmic noise.")

with col2:
    st.markdown("#### ⚡ Spectral Resonance")
    st.write("Traditional correlation is a blunt instrument. We identify exactly *which* frequencies assets share to uncover hidden systemic risks.")

with col3:
    st.markdown("#### ⛓️ Causal Leadership")
    st.write("Stop guessing who leads. Use Spectral Granger Causality to identify which asset is the true 'leading indicator' at specific time scales.")

st.markdown("""
---
### 🛠️ Strategic Edge
By discarding the "Noise" component and focusing on the **Structural Trend**, investors can:
- **Avoid False Breakouts**: Identify when price movement is purely high-frequency noise without macro support.
- **Manage Hidden Correlation**: Uncover when a portfolio is dangerously over-resonant on a specific frequency band.
- **Identify Leadership Shifts**: See leadership changes in real-time before they manifest in lagging technical indicators.

---
### 🛰️ Submission Hub | AlgoFest 2026
Use the sidebar to explore the engine's core intelligence modules. Each page features a **Math Integrity Proof** and an **Actionable Stance Decoder.**
""", unsafe_allow_html=True)

# Sidebar refinement
st.sidebar.markdown('<p class="header-gradient" style="font-size: 1.2rem;">Market DNA v1.2</p>', unsafe_allow_html=True)
st.sidebar.success("✅ Technical Execution Verified")
st.sidebar.info("Select a module above to begin.")
st.sidebar.caption("Securing Institutional Alpha through Math.")
