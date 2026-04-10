import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import compute_log_returns, normalize_signal
from engine.decompose import decompose_signal_mra, get_band_labels
from engine.scalogram import compute_cwt, compute_ssq_cwt, get_magnitude

st.set_page_config(page_title="Decomposition Explorer | FinSignal Suite", layout="wide")

st.title("🔍 Decomposition Explorer")

# Sidebar Controls
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Ticker", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD"], index=0)
wavelet_name = st.sidebar.selectbox("Wavelet Family", ["db4", "sym8", "coif1", "haar"], index=0)
level = st.sidebar.slider("Decomposition Depth", 2, 8, 5)

# Load Data
data = get_data(ticker)
if data is not None:
    returns = compute_log_returns(data)
    norm_returns = normalize_signal(returns)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"Raw Returns: {ticker}")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(y=returns, name="Log Returns", line=dict(color='rgba(100, 149, 237, 0.8)')))
        fig_raw.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_raw, use_container_width=True)

    # 1. DWT Decomposition
    st.divider()
    st.subheader("Multi-Resolution Analysis (DWT)")
    
    with st.spinner("Computing DWT..."):
        bands = decompose_signal_mra(norm_returns, wavelet=wavelet_name, level=level)
        labels = get_band_labels(level)
    
    fig_dwt = go.Figure()
    for i, (band, label) in enumerate(zip(bands, labels)):
        # Offset bands for visibility if needed, but stacked/separate is better
        # For this dashboard we'll overlay them with visibility controls
        visible = 'legendonly' if i > 1 else True
        fig_dwt.add_trace(go.Scatter(y=band, name=label, visible=visible))
    
    fig_dwt.update_layout(title="Frequency Band Decomposition (Additive Components)", height=500)
    st.plotly_chart(fig_dwt, use_container_width=True)

    # 2. CWT Scalogram
    st.divider()
    st.subheader("Continuous Wavelet Scalogram")
    
    cwt_type = st.radio("CWT Method", ["Standard CWT", "Synchrosqueezed CWT"])
    
    with st.spinner("Computing CWT (this may take a few seconds)..."):
        if cwt_type == "Standard CWT":
            Wx, scales = compute_cwt(norm_returns)
            magnitude = get_magnitude(Wx)
            y_axis_label = "Scale"
            y_values = scales
        else:
            Tx, Wx, ssq_freqs, scales = compute_ssq_cwt(norm_returns)
            magnitude = get_magnitude(Tx)
            y_axis_label = "Frequency"
            y_values = ssq_freqs
            
    fig_cwt = go.Figure(data=go.Heatmap(
        z=magnitude,
        x=np.arange(len(norm_returns)),
        y=y_values,
        colorscale='Viridis',
        showscale=False
    ))
    fig_cwt.update_layout(
        title=f"{cwt_type} Energy Distribution",
        xaxis_title="Time",
        yaxis_title=y_axis_label,
        height=500
    )
    st.plotly_chart(fig_cwt, use_container_width=True)
    
else:
    st.error("Could not load data for the selected ticker.")
