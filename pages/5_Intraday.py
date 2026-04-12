import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time as dtime
from engine.scalogram import track_frequency_flow, get_magnitude
from engine.intelligence import analyze_intraday, get_execution_playbook
from engine.ui import inject_custom_css

st.set_page_config(page_title="Intraday Spectral Flow | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Intraday Spectral Flow")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Real-Time Acquisition")
stock     = st.sidebar.selectbox("Intraday Symbol", ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "SPY", "QQQ", "BTC-USD"], index=0)
timeframe = st.sidebar.selectbox("Aggregation Window", ["1m", "5m", "15m"], index=0)

st.sidebar.divider()
st.sidebar.header("Processing Parameters")
stft_window = st.sidebar.slider("STFT Segment Length", 32, 256, 64, help="Spectral resolution vs temporal precision trade-off.")

# ── Load and Prep ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_intraday_data(symbol, interval):
    try:
        df = yf.download(symbol, period="5d", interval=interval)
        if df.empty: return None
        return df
    except Exception:
        return None

with st.spinner(f"Acquiring real-time vectors for {stock}..."):
    data = get_intraday_data(stock, timeframe)

if data is not None:
    # Use only the last 300 points for focus
    close_prices = data['Close'].tail(300).values.flatten()
    dates = data.index[-300:]
    
    # Normalize
    mean_p = np.mean(close_prices)
    std_p  = np.std(close_prices)
    norm_p = (close_prices - mean_p) / (std_p if std_p > 0 else 1.0)

    # ── 2. STFT Scalogram ──────────────────────────────────────────────────────
    with st.spinner("Calculating frequency flow:"):
        freqs, times, map_z = track_frequency_flow(norm_p, window_size=stft_window)
        intensity = get_magnitude(map_z)
        avg_energy = np.mean(intensity, axis=1)

        # High-Frequency Rhythm Detection
        dom_freq_idx = np.argmax(avg_energy)
        dom_freq = freqs[dom_freq_idx]
        time_res = int(timeframe[:-1]) if timeframe[-1] == 'm' else (60 if timeframe[-1] == 'h' else 1)
        dom_rhythm = (1.0 / dom_freq) * time_res if dom_freq > 0 else 0
        
        energy_peak = np.max(avg_energy)
        energy_mean = np.mean(avg_energy)
        compression = energy_peak / energy_mean if energy_mean > 0 else 1
        
        regime, pulse_force, description = analyze_intraday(compression, dom_rhythm)

    # ── 0. Strategy Analysis Matrix ──────────────────────────────
    st.subheader("Strategy Analysis Matrix")
    with st.expander("Primary Intraday Intelligence", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**PULSE REGIME**")
            st.header(regime)
            
            st.markdown("**RESONANCE FORCE**")
            st.progress(pulse_force)
            st.caption(f"Spectral Compression Ratio: {compression:.2f}x")
            
        with col2:
            st.markdown("**Analysis Methodology**")
            st.write(f"The intraday engine identifies a **{regime}** state. {description}")
            st.markdown(f"**Tactical Pulse**: Volatility is currently concentrated at the **~{dom_rhythm:.1f}-minute** heartbeat.")

            # Execution Playbook Injection
            st.markdown("**Execution Playbook**")
            playbook = get_execution_playbook("Intraday", regime)
            for step in playbook:
                st.write(step)

    # ── 1. Price Momentum ──────────────────────────────────────────────────────
    st.subheader(f"Price Vector: {stock} ({timeframe})")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dates, y=close_prices, name="Exec Price", line=dict(color='#4fa3e0', width=2)))
    fig_price.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        font=dict(family="JetBrains Mono"),
        xaxis=dict(gridcolor='#1a1d21'),
        yaxis=dict(gridcolor='#1a1d21'),
        xaxis_title="Temporal Sequence"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── 2. STFT Scalogram ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Short-Time Frequency Topology")
    
    with st.expander("Analytical Context", expanded=True):
        st.markdown(f"""
        **Short-Time Fourier Transform (STFT)** decomposes the intraday signal into localized frequency segments.
        - **Persistent Horizontal Lines**: Indicate continuous cyclical volatility at a specific frequency.
        - **Vertical Energy Bursts**: Signify impulsive regime shifts or news-driven volatility spikes.
        """)

    with st.spinner("Calculating frequency flow:"):
        freqs, times, map_z = track_frequency_flow(norm_p, window_size=stft_window)
        intensity = get_magnitude(map_z)

    fig_stft = go.Figure(data=go.Heatmap(
        z=np.log1p(intensity),
        x=np.arange(len(times)),
        y=freqs,
        colorscale='Magma',
        showscale=True,
        hovertemplate='<b>Segment:</b> %{x}<br><b>Freq:</b> %{y:.3f}<br><b>Intensity:</b> %{z:.3f}<extra></extra>'
    ))
    
    fig_stft.update_layout(
        title="Institutional STFT Scalogram (Volatility Distribution)",
        xaxis_title="Temporal Segments",
        yaxis_title="Frequency (Cycles/Interval)",
        height=450, margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        font=dict(family="JetBrains Mono"),
        xaxis=dict(gridcolor='#1a1d21'),
        yaxis=dict(gridcolor='#1a1d21')
    )
    st.plotly_chart(fig_stft, use_container_width=True)

    # ── 3. Heartbeat Audit Table (Cycle Attribution) ───────────────────────
    st.divider()
    st.subheader("Spectral Heartbeat Audit")
    
    # Extract top 3 rhythmic peaks
    peak_indices = np.argsort(avg_energy)[-5:][::-1]
    audit_data = []
    for idx in peak_indices:
        f = freqs[idx]
        period = (1.0 / f) * time_res if f > 0 else 0
        pwr = avg_energy[idx]
        audit_data.append({
            "Rhythm (Minutes)": f"~{period:.1f}m",
            "Frequency (Hz)": f"{f:.4f}",
            "Energy Power": f"{pwr:.4f}",
            "Status": "ACTIVE" if pwr > energy_mean * 2 else "STOCHASTIC"
        })
    st.table(pd.DataFrame(audit_data))
    st.caption("Clinical Audit: Heartbeat values represent localized temporal resonance in the intraday price vector.")

    # ── 4. Diagnostic Attribution (Technical) ──────────────────────────────
    st.divider()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Spectral Energy Distribution")
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=freqs, y=avg_energy, fill='tozeroy', line=dict(color='#00f0ff', width=2)))
        fig_psd.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_dark",
            font=dict(family="JetBrains Mono"),
            xaxis=dict(gridcolor='#1a1d21'),
            yaxis=dict(gridcolor='#1a1d21'),
            xaxis_title="Frequency Domain",
            yaxis_title="Energy Density"
        )
        st.plotly_chart(fig_psd, use_container_width=True)

    with c2:
        st.subheader("Technical Matrix")
        st.markdown(f"""
        | Metric | Clinical Value |
        |---|---|
        | Peak Heartbeat | `~{dom_rhythm:.1f} min` |
        | Energy Compression | `{compression:.2f}x` |
        | Nyquist Guard | `2 {timeframe}` |
        | Window Integrity | `{stft_window} units` |
        """)
        
        if compression > 4:
            st.error("Extreme energy compression detected.")
        elif compression > 2.5:
            st.warning("Elevated spectral density found.")
        else:
            st.success("Stable state confirmed.")

else:
    st.error("Data Acquisition Error: Check symbol connectivity or market trading hours.")
