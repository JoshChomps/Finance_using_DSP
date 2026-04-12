import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.coherence import calculate_coherence
from engine.intelligence import analyze_portfolio, get_execution_playbook
from engine.ui import inject_custom_css

st.set_page_config(page_title="Portfolio Resonance Matrix | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Portfolio Resonance Matrix")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Portfolio Construction")
default_tickers = ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD", "TSLA"]
tickers = st.sidebar.multiselect(
    "Asset Selection",
    options=default_tickers,
    default=["SPY", "QQQ", "GLD", "NVDA", "BTC-USD"],
    help="Select the institutional assets to include in the resonance mapping."
)

st.sidebar.divider()
st.sidebar.header("Analysis Parameters")
lookback = st.sidebar.slider("Analysis Horizon (Days)", 250, 1500, 750)

@st.cache_data(ttl=3600)
def load_portfolio_data(ticker_list, horizon):
    portfolio = {}
    for ticker in ticker_list:
        data = get_data(ticker)
        if data is not None:
            ret = calculate_returns(data).tail(horizon)
            portfolio[ticker] = z_score_normalize(ret).values
    return portfolio

# ── Load and Prep ──────────────────────────────────────────────────────────────
if len(tickers) < 2:
    st.info("Selection required: At least two assets for resonance mapping.")
else:
    with st.spinner("Analyzing spectral resonance..."):
        portfolio_data = load_portfolio_data(tickers, lookback)
        
    valid_tickers = list(portfolio_data.keys())
    
    if len(valid_tickers) < 2:
        st.error("Protocol Error: Insufficient data retrieved for the specified cluster.")
    else:
        # Calculate the Average Coherence Matrix
        n = len(valid_tickers)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    # Sync lengths if discordant
                    s1 = portfolio_data[valid_tickers[i]]
                    s2 = portfolio_data[valid_tickers[j]]
                    min_l = min(len(s1), len(s2))
                    coh, _, _, _, _ = calculate_coherence(s1[-min_l:], s2[-min_l:])
                    matrix[i, j] = np.mean(coh)
                    matrix[j, i] = matrix[i, j]
        
        avg_resonance = np.mean(matrix[np.triu_indices(n, k=1)])
        regime, sync_force, description = analyze_portfolio(avg_resonance)

        # ── 0. Strategy Analysis Matrix ──────────────────────────────
        st.subheader("Strategy Analysis Matrix")
        with st.expander("Primary Portfolio Intelligence", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**CONTAGION REGIME**")
                st.header(regime)
                
                st.markdown("**SYSTEMIC SYNC FORCE**")
                st.progress(sync_force)
                st.caption(f"Mean Portfolio Resonance: {avg_resonance:.4f}")
                
            with col2:
                st.markdown("**Analysis Methodology**")
                st.write(f"The structural engine identifies a **{regime}** state. {description}")
                
                # Execution Playbook Injection
                st.markdown("**Execution Playbook**")
                playbook = get_execution_playbook("Portfolio", regime)
                for step in playbook:
                    st.write(step)
                
                st.markdown(f"**Diversification Efficiency**: {max(0, min(100, 100 * (1 - avg_resonance))):.1f}%")

        # ── 1. Portfolio Construction Stats ─────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Mean Resonance", f"{avg_resonance:.4f}")
        with m2:
            risk_cat = "Elevated" if avg_resonance > 0.45 else ("Moderate" if avg_resonance > 0.25 else "Diversified")
            st.metric("Risk Profile", risk_cat)
        with m3:
            st.metric("Efficiency", f"{max(0, min(100, 100 * (1 - avg_resonance))):.1f}%")

        # ── 1. Resonance Topology ──────────────────────────────────────────────
        st.divider()
        col_map, col_rep = st.columns([1.8, 1])
        
        with col_map:
            st.subheader("Spectral Inter-dependency Matrix")
            fig_matrix = go.Figure(data=go.Heatmap(
                z=matrix,
                x=valid_tickers,
                y=valid_tickers,
                colorscale='Viridis',
                zmin=0, zmax=1,
                hovertemplate='<b>%{x} & %{y}</b><br>Resonance: %{z:.4f}<extra></extra>'
            ))
            fig_matrix.update_layout(
                height=550, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark",
                font=dict(family="JetBrains Mono"),
                xaxis=dict(gridcolor='#1a1d21'),
                yaxis=dict(gridcolor='#1a1d21')
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        with col_rep:
            st.subheader("Systemic Resonance Audit")
            
            # Identify top 5 most resonant pairs
            node_data = []
            for i in range(n):
                for j in range(i+1, n):
                    node_data.append({
                        "Asset Pair": f"{valid_tickers[i]} <> {valid_tickers[j]}",
                        "Resonance": matrix[i, j],
                        "Status": "CRITICAL" if matrix[i,j] > 0.6 else ("ALIGNED" if matrix[i,j] > 0.4 else "DECOUPLED")
                    })
            
            df_nodes = pd.DataFrame(node_data).sort_values("Resonance", ascending=False).head(5)
            st.table(df_nodes)
            
            st.markdown(f"""
            **Cluster Assessment:**
            Resonance Analysis for the `{lookback}-day` horizon indicates a **{risk_cat}** level of synchronization.
            
            - **Critical Nodes**: Pairs marked CRITICAL behave as a single macro-lever. 
            - **Orthogonal Buffers**: Decoupled nodes provide structural stability.
            """)
            
            if avg_resonance > 0.5:
                st.warning("Recommendation: Inject non-correlated (orthogonal) assets.")
            else:
                st.success("Stable state confirmed.")

        # ── 2. Focused Pair Deep-Dive ──────────────────────────────────────────
        st.divider()
        st.subheader("Focused Spectral Investigation")
        
        c_sel1, c_sel2 = st.columns(2)
        pair1 = c_sel1.selectbox("Asset Alpha", valid_tickers, index=0)
        pair2 = c_sel2.selectbox("Asset Beta", valid_tickers, index=min(1, len(valid_tickers)-1))
        
        if pair1 == pair2:
            st.info("Identity filter active: Pair-wise deep-dive requires two distinct assets.")
        else:
            s1 = portfolio_data[pair1]
            s2 = portfolio_data[pair2]
            min_l = min(len(s1), len(s2))
            coh, phase, coi, freqs, sig = calculate_coherence(s1[-min_l:], s2[-min_l:])
            
            # Period calculation
            periods = [1.0/f if f > 0 else 1000 for f in freqs]

            fig_detail = go.Figure(data=go.Heatmap(
                z=coh,
                x=np.arange(len(coh[0])),
                y=periods,
                colorscale='Inferno',
                zmin=0, zmax=1,
                hovertemplate='<b>Period:</b> %{y:.1f}d<br><b>Resonance:</b> %{z:.4f}<extra></extra>'
            ))
            fig_detail.update_layout(
                title=f"Detailed Cross-Spectral Distribution: {pair1} vs {pair2}",
                xaxis_title="Time Index (Horizon Segments)",
                yaxis_title="Cycle Period (Days)",
                yaxis_type="log",
                yaxis=dict(autorange="reversed", gridcolor='#1a1d21'),
                xaxis=dict(gridcolor='#1a1d21'),
                height=450, margin=dict(l=0, r=0, t=30, b=0),
                template="plotly_dark",
                font=dict(family="JetBrains Mono")
            )
            st.plotly_chart(fig_detail, use_container_width=True)
            
            st.caption(f"Clinical Audit: Resonance between {pair1} and {pair2} is currently concentrated at the structural frequencies. Consult the Bivariate Guardian for phase-lead attribution.")

st.sidebar.caption("Institutional-grade Alpha through Digital Signal Processing.")
