import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.coherence import calculate_coherence
from engine.ui import inject_custom_css

st.set_page_config(page_title="Portfolio Resonance Guardian | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Resonance Guardian")
st.markdown("Identification of systemic risk through multi-asset spectral resonance analysis.")

# Sidebar for Portfolio Input
st.sidebar.header("Portfolio Construction")
default_tickers = ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD", "TSLA"]
tickers = st.sidebar.multiselect(
    "Asset Selection",
    options=default_tickers,
    default=["SPY", "QQQ", "TSLA", "AAPL", "BTC-USD"],
    help="Select the institutional assets to include in the resonance mapping."
)

st.sidebar.divider()
lookback = st.sidebar.slider("Analysis Horizon (Days)", 250, 1000, 500)

@st.cache_data(ttl=3600)
def load_portfolio_data(ticker_list):
    portfolio = {}
    for ticker in ticker_list:
        data = get_data(ticker)
        if data is not None:
            ret = calculate_returns(data).tail(lookback)
            portfolio[ticker] = z_score_normalize(ret).values
    return portfolio

if len(tickers) < 2:
    st.warning("Selection required: at least two assets for resonance mapping.")
else:
    with st.spinner("Analyzing portfolio resonance:"):
        portfolio_data = load_portfolio_data(tickers)
        
    valid_tickers = list(portfolio_data.keys())
    
    if len(valid_tickers) < 2:
        st.error("Data retrieval failed for the specified assets.")
    else:
        # Calculate the Average Coherence Matrix
        n = len(valid_tickers)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    coh, phase, coi, freqs, sig = calculate_coherence(portfolio_data[valid_tickers[i]], portfolio_data[valid_tickers[j]])
                    # Use the mean coherence across all frequencies and time as the "Resonance Score"
                    score = np.mean(coh)
                    matrix[i, j] = score
                    matrix[j, i] = score
        
        avg_resonance = np.mean(matrix[np.triu_indices(n, k=1)])
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Mean Portfolio Resonance", f"{avg_resonance:.4f}")
        with m2:
            risk_level = "High Correlation" if avg_resonance > 0.4 else "Diversified"
            st.metric("Systemic Risk Profile", risk_level)
        with m3:
            diversity_health = max(0, 100 - (avg_resonance * 150))
            st.metric("Diversity Efficiency Score", f"{diversity_health:.1f}/100")

        st.divider()

        # 1. Resonance Matrix (Heatmap)
        col_m, col_t = st.columns([2, 1])
        
        with col_m:
            st.subheader("Cross-Spectral Correlation Matrix")
            fig_matrix = go.Figure(data=go.Heatmap(
                z=matrix,
                x=valid_tickers,
                y=valid_tickers,
                colorscale='Viridis',
                zmin=0, zmax=1,
                hovertemplate='<b>%{x} & %{y}</b><br>Resonance: %{z:.4f}<extra></extra>'
            ))
            fig_matrix.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_matrix, width='stretch')
        
        with col_t:
            st.subheader("Analytical Report")
            with st.expander("Spectral Resonance Methodology", expanded=True):
                st.markdown("""
                Traditional correlation measurements provide an aggregate view of asset alignment. **Spectral Resonance** decomposes this alignment into specific frequency components.
                
                Assets may exhibit decoupling at high frequencies while maintaining high resonance at **low (macro) frequencies**. 
                
                **Matrix Interpretation:**
                - **Intensity**: High values indicate synchronized movement across the spectral domain.
                - **Threshold ( > 0.5)**: Signifies increased exposure to synchronized systemic shocks.
                - **Lower Bound ( < 0.2)**: Demonstrates high statistical diversification.
                """)
            
            if avg_resonance > 0.4:
                st.error("Risk Alert: Elevated portfolio resonance detected. Systemic macro-frequency shocks may impact holdings simultaneously.")
            else:
                st.success("Diversification Verified: Low cross-spectral resonance confirmed across the analyzed horizon.")

        st.divider()
        
        # 2. Resonance Over Time (Manual Comparison)
        st.subheader("Detailed Resonance Investigation")
        
        c_sel1, c_sel2 = st.columns(2)
        pair1 = c_sel1.selectbox("Asset A", valid_tickers, index=0)
        pair2 = c_sel2.selectbox("Asset B", valid_tickers, index=min(1, len(valid_tickers)-1))
        
        if pair1 == pair2:
            st.warning("Identity relationship: Select two distinct assets for spectral comparison.")
        else:
            coh, phase, coi, freqs, sig = calculate_coherence(portfolio_data[pair1], portfolio_data[pair2])
            
            fig_detail = go.Figure(data=go.Heatmap(
                z=coh,
                y=freqs,
                colorscale='Inferno',
                hovertemplate='<b>Freq:</b> %{y:.3f}<br><b>Resonance:</b> %{z:.4f}<extra></extra>'
            ))
            fig_detail.update_layout(
                title=f"Detailed Cross-Spectral distribution: {pair1} vs {pair2}",
                xaxis_title="Timeline Segments",
                yaxis_title="Frequency",
                height=400,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_detail, width='stretch')
            
            st.info(f"Insight: {pair1} and {pair2} exhibit localized coherence at the identified frequencies. Portfolio management should account for shared spectral characteristics between these assets.")

st.sidebar.caption("Institutional Alpha through Digital Signal Processing.")
