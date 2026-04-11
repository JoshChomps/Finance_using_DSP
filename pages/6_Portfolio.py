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

st.markdown('<h1 class="header-gradient">Resonance Guardian</h1>', unsafe_allow_html=True)
st.markdown("### Detect hidden systemic risk in your diversification strategy.")

# Sidebar for Portfolio Input
st.sidebar.header("Portfolio Construction")
tickers_input = st.sidebar.text_input("Enter Ticker Symbols (comma separated)", value="SPY, QQQ, TSLA, AAPL, BTC-USD")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

st.sidebar.markdown("---")
lookback = st.sidebar.slider("Analysis Window (Days)", 250, 1000, 500)

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
    st.warning("Please enter at least two tickers to analyze resonance.")
else:
    with st.spinner("Analyzing portfolio resonance..."):
        portfolio_data = load_portfolio_data(tickers)
        
    valid_tickers = list(portfolio_data.keys())
    
    if len(valid_tickers) < 2:
        st.error("Not enough data found for the provided tickers.")
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
        
        # ── Top Intelligence Bar ───────────────────────────────────────────────
        avg_resonance = np.mean(matrix[np.triu_indices(n, k=1)])
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Avg Portfolio Resonance", f"{avg_resonance:.4f}")
        with m2:
            risk_level = "High Correlation" if avg_resonance > 0.4 else "Diversified"
            st.metric("Systemic Risk Level", risk_level)
        with m3:
            # Score out of 100
            diversity_health = max(0, 100 - (avg_resonance * 150))
            st.metric("Diversity Health Score", f"{diversity_health:.1f}/100")

        st.divider()

        # 1. Resonance Matrix (Heatmap)
        col_m, col_t = st.columns([2, 1])
        
        with col_m:
            st.subheader("Correlation Spectrum Matrix")
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
            st.subheader("Intelligence Report")
            with st.expander("📖 What is Hidden Resonance?", expanded=True):
                st.markdown("""
                Traditional correlation only tells you if two stocks move in the same direction. **Resonance** tells you if they share the same *frequency DNA*. 
                
                Even if two stocks look uncorrelated in the short term, they might be highly resonant at a **Macro frequency**. 
                
                **How to read the Matrix:**
                - **The Brighter the Spot**: The more "in sync" these two assets are. 
                - **High Resonance (>0.5)**: Your portfolio is dangerously exposed to the same structural shocks.
                - **Low Resonance (<0.2)**: True mathematical diversification.
                """)
            
            if avg_resonance > 0.4:
                st.error("⚠️ **Risk Alert**: Your portfolio assets are highly resonant. A single macro-frequency shock could impact most of your holdings simultaneously.")
            else:
                st.success("✅ **Diversification Verified**: Your portfolio shows low cross-spectral resonance.")

        st.divider()
        
        # 2. Resonance Over Time (Focus on the most resonant pair)
        if n >= 2:
            # Find the most resonant pair (excluding diagonal)
            flat_idx = np.argmax(np.triu(matrix, k=1))
            row_idx, col_idx = np.unravel_index(flat_idx, matrix.shape)
            pair1, pair2 = valid_tickers[row_idx], valid_tickers[col_idx]
            
            st.subheader(f"Deep-Dive: Most Resonant Pair ({pair1} & {pair2})")
            
            coh, phase, coi, freqs, sig = calculate_coherence(portfolio_data[pair1], portfolio_data[pair2])
            
            fig_detail = go.Figure(data=go.Heatmap(
                z=coh,
                y=freqs,
                colorscale='Inferno',
                hovertemplate='<b>Freq:</b> %{y:.3f}<br><b>Resonance:</b> %{z:.4f}<extra></extra>'
            ))
            fig_detail.update_layout(
                title=f"Cross-Spectral mapping for {pair1} vs {pair2}",
                xaxis_title="Timeline Blocks",
                yaxis_title="Frequency",
                height=400,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_detail, width='stretch')
            
            st.info(f"💡 **Insight**: {pair1} and {pair2} share significant energy at the frequencies shown in orange/white. Avoid adding more assets with similar characteristics to maintain your Diversity Health.")

st.sidebar.caption("Securing Institutional Alpha through Math.")
