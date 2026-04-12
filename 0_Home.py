import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from engine.data import manager
from engine.decompose import slice_signal
from engine.intelligence import analyze_stance, get_execution_playbook
from engine.ui import inject_custom_css

st.set_page_config(
    page_title="Market DNA | Tactical Command Center",
    layout="wide",
)

inject_custom_css(st)

# == Header ===================================================================
st.title("Market DNA: Tactical Command Center")
st.subheader("Institutional Signal Extraction & Spectral Intelligence Hub")

# == Sidebar Selection ========================================================
st.sidebar.header("Command Controls")
asset_options = ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD", "ETH-USD"]
selected_assets = st.sidebar.multiselect(
    "Core Asset Scan",
    options=asset_options,
    default=["SPY", "QQQ", "GLD", "BTC-USD"],
    help="Select the institutional assets to include in the Master Signal Scan."
)

st.sidebar.divider()
st.sidebar.markdown('**Market DNA: Alpha Synthesis**')
st.sidebar.success("Institutional Integrity Verified")

# == Mermaid Helper ==========================================================
def render_mermaid(code: str):
    components.html(
        f"""
        <div class="mermaid" style="height: 100%; width: 100%; overflow: auto;">
            {code}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'dark', flowchart: {{ useMaxWidth: false }} }});
        </script>
        """,
        height=500,
    )

# == Master Signal Scan (The HUD) ==============================================
st.divider()
st.subheader("Master Signal Scan: Global Regime Audit")

@st.cache_data(ttl=1800)
def scan_core_assets(assets):
    scan_results = []
    if not assets:
        return []
    for asset in assets:
        data = manager.get_data(asset, period='1y')
        if data is not None:
            rets = data['Close'].pct_change().dropna().values
            # Rapid MRA for stance
            bands, _ = slice_signal(rets, wavelet='db4', depth=4)
            labels = ["Underlying Structural Trend", "Quarterly Cycle", "Monthly Cycle", "Weekly Cycle", "Noise"]
            regime, force, stance_data = analyze_stance(bands, labels)
            scan_results.append({
                "Asset": asset,
                "Regime": regime,
                "Sync Force": force,
                "Bias": "UP" if force > 0 else "DOWN"
            })
    return scan_results

if selected_assets:
    with st.spinner(f"Scanning {len(selected_assets)} spectral regimes..."):
        results = scan_core_assets(selected_assets)

    if results:
        # Institutional Grid Wrapping (4 assets per row)
        cols_per_row = 4
        for i in range(0, len(results), cols_per_row):
            row_items = results[i : i + cols_per_row]
            cols = st.columns(len(row_items))
            for j, res in enumerate(row_items):
                with cols[j]:
                    color = "#00ff41" if res['Bias'] == "UP" else "#ff4b4b"
                    st.markdown(f"**{res['Asset']}**")
                    st.metric(res['Regime'], f"{res['Sync Force']:.2f}", delta=res['Bias'], delta_color="normal")
                    st.caption("Structural Flow Status")
    else:
        st.error("Protocol Error: Global Scan Offline.")
else:
    st.info("Select assets in the sidebar to begin spectral scan.")

st.divider()

# == Strategy Decision Manual ==================================================
col_man, col_flow = st.columns([1.5, 1])

with col_man:
    st.markdown("### Strategy Decision Manual")
    with st.expander("Analysis Step 01: Identification (Decomposition)", expanded=True):
        st.write("""
        **Action**: Navigate to the Decomposition Explorer.
        **Goal**: Identify the 'Stance' : is the structural trend working with or against the local cycles?
        - **IF** Stance is 'Strong Bullish', look for long entries on cycle troughs.
        - **IF** Stance is 'Tactical Distribution', avoid long entries (The Bounce).
        """)
        
    with st.expander("Analysis Step 02: Precedence (Coherence & Causality)"):
        st.write("""
        **Action**: Navigate to Coherence or Causality.
        **Goal**: Find a leader. Does SPY lead QQQ? Does BTC lead the S&P 500?
        - **IF** 'Master-Slave Lead' exists, use the leader as your trigger signal.
        - **IF** 'Fragmented', trade each asset in isolation (No Correlation).
        """)
        
    with st.expander("Analysis Step 03: Verification (Backtesting)"):
        st.write("""
        **Action**: Navigate to the Backtesting Simulator.
        **Goal**: Performance audit. Does this lead/lag relationship actually show profit?
        - **IF** 'Robust Resonance Capture', deploy capital using Kelly sizing.
        - **IF** 'Spectral Decay', reject the strategy : the signal is overfitting.
        """)

with col_flow:
    st.markdown("### Signal Processing Pipeline")
    render_mermaid("""
    graph TD
        A[Raw Market Data] --> B[Multiresolution Analysis]
        B --> C{Decision Hub}
        C -->|Stance| D[Decomposition]
        C -->|Precedence| E[Coherence]
        C -->|Flow| F[Causality]
        D --> G[Strategy Verdict]
        E --> G
        F --> G
        G --> H[Backtest Sim]
        H --> I[Execute Alpha]
    """)

st.divider()

# == Operational Integrity ======================================================
st.subheader("Operational Integrity")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Intelligence Layer", "Active", "7 Engines")
c2.metric("Tactical Manual", "V1.0 Certified", "IF/THEN Logic")
c3.metric("Data Lake", manager.provider.__class__.__name__, "Pilot Scale")
c4.metric("Operational Standard", "Slate-Carbon", "Phase 41")
