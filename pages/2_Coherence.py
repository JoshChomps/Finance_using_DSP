import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import compute_log_returns, normalize_signal
from engine.coherence import compute_wavelet_coherence

st.set_page_config(page_title="Cross-Asset Resonance | FinSignal Suite", layout="wide")

st.title("🤝 Cross-Asset Resonance")

# Sidebar Controls
st.sidebar.header("Settings")
t1 = st.sidebar.selectbox("First Ticker", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=0)
t2 = st.sidebar.selectbox("Second Ticker", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=2) # Default GLD

if t1 == t2:
    st.warning("Please select two different tickers for coherence analysis.")
else:
    # Load Data
    d1 = get_data(t1)
    d2 = get_data(t2)
    
    if d1 is not None and d2 is not None:
        # Align data (truncate to shortest)
        min_len = min(len(d1), len(d2))
        r1 = compute_log_returns(d1).tail(min_len)
        r2 = compute_log_returns(d2).tail(min_len)
        
        # Normalize
        r1_n = normalize_signal(r1)
        r2_n = normalize_signal(r2)
        
        st.subheader(f"Analyzing {t1} vs {t2}")
        
        with st.spinner("Computing Wavelet Coherence (this is intensive)..."):
            # Limit data to last 1000 points to keep it responsive in the dashboard
            max_points = 750
            y1 = r1_n.tail(max_points).values
            y2 = r2_n.tail(max_points).values
            
            wct, phase, coi, freqs, sig = compute_wavelet_coherence(y1, y2)
            
        # 1. Coherence Heatmap
        fig_wct = go.Figure(data=go.Heatmap(
            z=wct,
            x=np.arange(len(y1)),
            y=freqs,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="Coherence")
        ))
        
        # Overlay Cone of Influence
        # coi is the edge of the reliable region
        fig_wct.add_trace(go.Scatter(
            x=np.arange(len(y1)),
            y=1.0/coi, # coi usually in period, converting to freq if needed or just showing period
            name="Cone of Influence",
            line=dict(color='white', dash='dash'),
            showlegend=False
        ))

        fig_wct.update_layout(
            title=f"Wavelet Coherence: {t1} vs {t2}",
            xaxis_title="Time (Days)",
            yaxis_title="Frequency",
            height=600
        )
        st.plotly_chart(fig_wct, use_container_width=True)
        
        # 2. Insights
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            avg_coherence = np.mean(wct, axis=1)
            fig_avg = go.Figure()
            fig_avg.add_trace(go.Bar(x=freqs, y=avg_coherence))
            fig_avg.update_layout(title="Average Coherence per Frequency", xaxis_title="Frequency", yaxis_title="Mean WCT")
            st.plotly_chart(fig_avg, use_container_width=True)
            
        with col2:
            st.markdown(f"""
            #### How to interpret this chart:
            - **Bright Red/Yellow Areas**: High resonance. The two assets are tightly coupled at these frequencies and times.
            - **Dark Areas**: Low resonance. The assets are moving independently.
            - **V-Shape (Cone of Influence)**: Data outside this dashed line may be affected by edge artifacts.
            
            **Current Context:**
            - Average Coherence: `{np.mean(wct):.4f}`
            - Peak Coherence: `{np.max(wct):.4f}`
            """)
            
    else:
        st.error("Error loading data for one or both tickers.")
