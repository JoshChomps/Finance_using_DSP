import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.decompose import slice_signal, create_labels
from engine.scalogram import run_cwt_analysis, run_synchrosqueezing, get_magnitude
from engine.ui import inject_custom_css

st.set_page_config(page_title="Decomposition Explorer | FinSignal Suite", layout="wide")
inject_custom_css(st)

st.title("Decomposition Explorer")

# Sidebar Controls
st.sidebar.header("Analysis Settings")
symbol = st.sidebar.selectbox("Asset Symbol", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD"], index=0)
wavelet_name = st.sidebar.selectbox("Wavelet Family", ["db4", "sym8", "coif1", "haar"], index=0)
depth = st.sidebar.slider("Analysis Depth", 2, 8, 5)

# Load Data
data = get_data(symbol)
if data is not None:
    returns = calculate_returns(data)
    norm_data = z_score_normalize(returns)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"Price Momentum: {symbol}")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(y=returns, name="Daily Returns", line=dict(color='rgba(100, 149, 237, 0.8)')))
        fig_raw.update_layout(
            height=300, 
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # 1. Multi-Resolution Analysis
    st.divider()
    st.subheader("Underlying Market Cycles (MRA)")
    with st.expander("📖 What does this mean?", expanded=True):
        st.markdown(f"""
        **MRA (Multi-Resolution Analysis)** literally separates the fast 'noise' of the market from the deep, slow-moving 'structural' trends. 
        - **D1 / D2 (Top Lines)**: Fast-paced, high-frequency noise. Often mean-reverting.
        - **D3 / D4 (Middle Lines)**: Multi-day to weekly momentum shifts. Good for swing trading.
        - **A (Bottom Line)**: The smoothed, underlying macroeconomic trend. If `A` is pointing up, {symbol} is in a structural bull market regardless of daily red candles.
        """)
    
    with st.spinner("Breaking down the signal..."):
        bands = slice_signal(norm_data, wavelet=wavelet_name, depth=depth)
        band_names = create_labels(depth)
    
    fig_dwt = go.Figure()
    for i, (band, name) in enumerate(zip(bands, band_names)):
        # Display the most important cycles by default
        is_visible = 'legendonly' if i > 1 else True
        fig_dwt.add_trace(go.Scatter(y=band, name=name, visible=is_visible))
    
    fig_dwt.update_layout(
        title="Cycle Decomposition (Total Signal Pieces)", 
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_dwt, use_container_width=True)

    # 2. CWT Scalogram
    st.divider()
    st.subheader("Time-Frequency Energy Heatmap")
    
    with st.expander("📖 How to read this Heatmap", expanded=True):
        st.markdown(f"""
        This scalogram shows where the **energy (volatility)** of {symbol} is concentrated at any given point in time. 
        - **Y-Axis (Frequency/Scale)**: Lower scale = fast intra-week volatility. Higher scale = slow macro volatility.
        - **X-Axis (Time)**: The historical timeline.
        - **Bright Yellow/Red Spots**: Intense bursts of market energy. A wide vertical burst means volatility is shocking the system across *all* timeframes (classic crash signature).
        """)
    
    method = st.radio("Transform Method", ["Adaptive Wavelet", "Synchrosqueezed (Sharper)"])
    
    with st.spinner("Generating heatmap..."):
        if method == "Adaptive Wavelet":
            map_data, scales = run_cwt_analysis(norm_data)
            intensity = get_magnitude(map_data)
            y_label = "Cycle Scale"
            y_axis = scales
        else:
            tight_map, raw_map, ssq_freqs, scales = run_synchrosqueezing(norm_data)
            intensity = get_magnitude(tight_map)
            y_label = "Relative Frequency"
            y_axis = ssq_freqs
            
    fig_heat = go.Figure(data=go.Heatmap(
        z=intensity,
        x=np.arange(len(norm_data)),
        y=y_axis,
        colorscale='Viridis',
        showscale=False
    ))
    fig_heat.update_layout(
        title=f"{method} Energy distribution over time",
        xaxis_title="Time index",
        yaxis_title=y_label,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
else:
    st.error("We couldn't pull the data for that symbol.")

