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

st.title("🔍 Decomposition Explorer")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Analysis Settings")
symbol       = st.sidebar.selectbox("Asset Symbol", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD"], index=0)
wavelet_name = st.sidebar.selectbox("Wavelet Family", ["db4", "sym8", "coif1", "haar"], index=0)
depth        = st.sidebar.slider("Decomposition Depth", 2, 8, 5)

# ── Load & Prep ────────────────────────────────────────────────────────────────
data = get_data(symbol)
if data is not None:
    returns   = calculate_returns(data)
    norm_data = z_score_normalize(returns)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(f"Daily Returns: {symbol}")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            y=returns.values, name="Daily Returns",
            line=dict(color="rgba(100, 149, 237, 0.8)"),
        ))
        fig_raw.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_raw, use_container_width=True)

    # ── 1. Multi-Resolution Analysis ──────────────────────────────────────────
    st.divider()
    st.subheader("Market Cycles — Multiresolution Analysis")

    with st.spinner("Decomposing signal..."):
        bands      = slice_signal(norm_data, wavelet=wavelet_name, depth=depth)
        band_names = create_labels(depth)

    # Find the non-trend band with the most energy (most active cycle)
    cycle_bands  = bands[1:]          # exclude the macro trend (approx)
    cycle_names  = band_names[1:]
    band_energy  = [np.var(b) for b in cycle_bands]
    dominant_idx = int(np.argmax(band_energy))

    st.info(
        f"Dominant cycle: **{cycle_names[dominant_idx]}** "
        f"(highest energy — this is the most active market rhythm in `{symbol}` "
        f"at this wavelet depth)"
    )

    fig_dwt = go.Figure()
    for i, (band, name) in enumerate(zip(bands, band_names)):
        # Show the macro trend and dominant cycle by default; others collapsed
        is_dominant = (i > 0 and i - 1 == dominant_idx)
        visible     = True if (i == 0 or is_dominant) else "legendonly"
        width       = 2 if is_dominant else 1
        fig_dwt.add_trace(go.Scatter(
            y=band, name=name, visible=visible,
            line=dict(width=width),
        ))

    fig_dwt.update_layout(
        title=f"Cycle Decomposition ({symbol})",
        yaxis_title="Normalized Signal",
        xaxis_title="Trading Days",
        height=500,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_dwt, use_container_width=True)

    # Period energy summary table
    energy_pct = np.array(band_energy) / max(sum(band_energy), 1e-12) * 100
    energy_df  = pd.DataFrame({
        "Cycle Band":       cycle_names,
        "Energy Share (%)": [f"{e:.1f}%" for e in energy_pct],
    })
    st.dataframe(energy_df, hide_index=True, use_container_width=True)

    # ── 2. Time-Frequency Heatmap ─────────────────────────────────────────────
    st.divider()
    st.subheader("Time-Frequency Energy Map")

    method = st.radio(
        "Transform Method",
        ["Adaptive Wavelet (CWT)", "Synchrosqueezed (Sharper)"],
        horizontal=True,
    )

    with st.spinner("Generating heatmap..."):
        if method == "Adaptive Wavelet (CWT)":
            map_data, scales = run_cwt_analysis(norm_data)
            intensity = get_magnitude(map_data)
            y_label   = "Scale"
            y_axis    = scales
        else:
            tight_map, _, ssq_freqs, scales = run_synchrosqueezing(norm_data)
            intensity = get_magnitude(tight_map)
            y_label   = "Frequency (cycles/day)"
            y_axis    = ssq_freqs

    fig_heat = go.Figure(data=go.Heatmap(
        z=np.log1p(intensity),          # log scale makes lower-energy bands visible
        x=np.arange(len(norm_data)),
        y=y_axis,
        colorscale="Viridis",
        showscale=False,
    ))
    fig_heat.update_layout(
        title=f"{method} — energy concentration over time ({symbol})",
        xaxis_title="Trading Days",
        yaxis_title=y_label,
        height=500,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption(
        "The heatmap shows **where energy is concentrated** across time and frequency. "
        "Bright streaks at a constant period indicate a persistent market cycle. "
        "Log-scaled intensity so both strong and weak cycles are visible."
    )

else:
    st.error("We couldn't pull data for that symbol.")
