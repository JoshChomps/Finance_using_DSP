import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from engine.data import get_data
from engine.decompose import slice_signal
from engine.intelligence import analyze_stance, forecast_spectral_path, get_execution_playbook
from engine.ui import inject_custom_css

st.set_page_config(page_title="Decomposition Explorer | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Decomposition Explorer")

with st.expander("Institutional Glossary | MRA Methodology", expanded=False):
    st.markdown("""
    **Prerequisites for Alpha Extraction**:
    - **MRA (Multiresolution Analysis)**: The decomposition of a signal into orthogonal frequency bands. Each band represents a specific 'Market Rhythm'.
    - **Structural Trend**: The underlying macro bias (Approximation). It filters out all stochastic noise to find the path of least resistance.
    - **Price Vector (Projection)**: A first-order structural extrapolation. It identifies where the core trend ('DNA') is pointing over a 14-day horizon.
    - **Dominant Cycle**: The frequency band currently containing the highest signal energy (variance). This is the 'heartbeat' of current price volatility.
    - **Accumulation**: A regime where the Structural Trend is positive, but intermediate cycles are corrective (The Dip). Historically an institutional buying zone.
    """)

# Sidebar
st.sidebar.header("Analysis Settings")
symbol       = st.sidebar.selectbox("Asset Symbol", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD"], index=0)
wavelet_name = st.sidebar.selectbox("Wavelet Type", ["db4", "sym8", "dmey"], index=1, 
                                   help="""
                                   **Wavelet Selection Logic**:
                                   - **db4 (Daubechies)**: High temporal precision. Best for detecting volatility spikes and sharp regime changes.
                                   - **sym8 (Symlets)**: Highest phase symmetry. Best for analyzing cyclic momentum and swing persistence without timing distortion.
                                   - **dmey (Meyer)**: Pure frequency resolution. Best for deep macro-structural analysis over very long lookbacks.
                                   """)
depth        = st.sidebar.slider("Decomposition Depth", 4, 10, 8)

data = get_data(symbol)

if data is not None:
    prices = data["Close"].values
    dates  = data.index
    
    with st.spinner("Extracting Spectral DNA..."):
        # Correct return: bands (list), actual_depth (int)
        bands, actual_depth = slice_signal(prices, wavelet=wavelet_name, depth=depth)
        
        # PyWavelets MRA returns: [Approximation, Detail_J, Detail_J-1, ..., Detail_1]
        # Approximation is the Structural Trend (Macro)
        # Detail_1 is the highest frequency noise (Micro)
        
        # Calculate Energy Share for Dominant Cycle Detection 
        band_energy = [np.var(b) for b in bands]
        
        # Pick dominant cycle from detail bands only (indices 1 to last)
        dominant_detail_idx = int(np.argmax(band_energy[1:])) + 1
        
        # Standardized Band/Cycle naming and period estimation (Coarse-to-Fine)
        n_details = len(bands) - 1
        periods = [500] + [2**(actual_depth - i + 1) for i in range(n_details)]
        band_names = ["Underlying Structural Trend"] + [f"Scale {actual_depth-i} (~{periods[i+1]}d)" for i in range(n_details)]
        
    # Analysis Stance
    stance_label, score, stance_data = analyze_stance(bands, band_names)
    
    # Phase 23: Spectral Projection (Price Vector) - horizon 14 days
    vector = forecast_spectral_path(bands[0], bands[dominant_detail_idx], horizon=14)
    # Align projection dates with future timeframe
    future_dates = pd.date_range(start=dates[-1], periods=15, freq=data.index.freq or 'D')[1:]

    # ── Top-Level Intelligence Dashboard ──────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Spectral Stance: {symbol}")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=dates, y=prices, name="Raw Price", line=dict(color="rgba(255,255,255,0.3)")))
        fig_raw.add_trace(go.Scatter(x=dates, y=bands[0], name="Structural Trend", line=dict(color="#00ff41", width=3)))
        
        # Phase 23 Projection Path
        fig_raw.add_trace(go.Scatter(
            x=future_dates, y=vector, 
            name="Spectral Forecast (T+14)", 
            line=dict(color="#00ff41", dash='dot', width=2),
            hovertemplate='<b>Forecast:</b> %{y:.2f}<extra></extra>'
        ))
        
        fig_raw.update_layout(
            title="Price vs. Spectral Forecast (Structural + Cycle)",
            height=400, margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_dark",
            font=dict(family="JetBrains Mono"),
            xaxis=dict(gridcolor='#1a1d21', title="Time"),
            yaxis=dict(gridcolor='#1a1d21', title="Price (USD)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    with col2:
        st.subheader("Dynamic Health Check")
        for s in stance_data[:4]: # Show top 4 influential bands
            color = "#00ff41" if s['direction'] == "UP" else "#ff4b4b"
            st.markdown(f"**{s['name']}**")
            st.markdown(f"<p style='color:{color}; font-size:1.2rem; margin-top:-10px;'>{s['direction']} (Strength: {s['strength']:.2f})</p>", unsafe_allow_html=True)
            st.progress(float(np.clip(s['strength']/2, 0.0, 1.0)))

    # 1. Strategy Analysis Matrix
    st.subheader("Strategy Analysis Matrix")
    with st.expander("Primary Signal Intelligence", expanded=True):
        col_st, col_tx = st.columns([1, 2])
        with col_st:
            st.metric("Current Stance", stance_label, help="The composite stance across all spectral scales.")
            
            # Phase 23 Target Metric
            target_price = vector[-1]
            price_delta = target_price - prices[-1]
            st.metric("Projected Target (T+14)", f"${target_price:.2f}", f"{price_delta:+.2f}")
            
            st.write(f"**Asset**: {symbol}")
            st.progress(float(np.clip((score + 0.5) / 1.0, 0.0, 1.0)))
            st.caption("Score indicates structural alignment (0.5 = Pure Momentum)")
            
        with col_tx:
            # Semantic Logic Generation
            is_bullish = score > 0
            logic_mode = "Clinical Aggression" if abs(score) > 0.3 else "Tactical Observation"
            
            st.markdown(f"""
            **Analysis Methodology**:
            The engine detects a **{stance_label}** regime. This is driven primarily by the `{band_names[-1]}`.
            
            **Execution Playbook**:
            """)
            playbook = get_execution_playbook("Decomposition", stance_label)
            for step in playbook:
                st.write(step)
            
            st.markdown(f"""
            **Predictive Horizon (T+14)**:
            The model manifests a synthetic price target of **${target_price:.2f}**. This represents the 
            expected structural settling point after accounting for current cyclical phase shifts.
            
            **Tactical Action**:
            The model suggests a **{logic_mode}** approach. The current structural DNA favors 
            **{'Institutional Accumulation' if is_bullish else 'Structural Distribution'}** 
            until the target horizon is reached.
            
            **Dominant Cycle Attribution**:
            The **{band_names[dominant_detail_idx]}** contains the most structural energy mass (excluding drift). 
            The market is currently oscillating most heavily at a **~{periods[dominant_detail_idx]}-day** rhythm. 
            Strategy execution should be tuned to this frequency for optimal resonance.
            """)

    # 1.1 Spectral Signal Audit (Actionable Data Table)
    st.markdown("#### Spectral Signal Audit")
    
    # Construct actionable signal dataframe
    audit_rows = []
    for i, s in enumerate(stance_data):
        m_score = s['score'] / s['weight'] # Normalized momentum [-1, 1]
        
        # Action Logic Mapping
        if m_score > 0.4: stance = "STRONG ACCUMULATE"
        elif m_score > 0.1: stance = "ACCUMULATE"
        elif m_score < -0.4: stance = "STRONG TRIM"
        elif m_score < -0.1: stance = "STRATEGIC TRIM"
        else: stance = "NEUTRAL / HOLD"
        
        audit_rows.append({
            "Spectral Band": s['name'],
            "Target Rhythm (Days)": f"~{periods[i]}d",
            "Momentum Force": round(m_score, 3),
            "Alignment": s['direction'],
            "Tactical Stance": stance
        })
    
    df_audit = pd.DataFrame(audit_rows)
    st.dataframe(
        df_audit,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Momentum Force": st.column_config.NumberColumn(format="%.3f"),
            "Tactical Stance": st.column_config.TextColumn(help="Standardized institutional action based on spectral momentum.")
        }
    )

    st.divider()

    # 2. Multi-Resolution Analysis (Bimodal Delivery)
    st.subheader("Spectral Decomposition (Multiresolution Audit)")
    with st.expander("Methodology Summary", expanded=False):
        st.markdown("""
        **MRA (Multi-Resolution Analysis)** separates market noise from deep, structural trends using orthogonal wavelets.
        - **Structural Layer**: The low-frequency anchor of the price action.
        - **Harmonic Layer**: Centered-on-zero rhythmic cycles representing harmonic volatility.
        """)

    # 2A. Structural DNA View (Trend + Forecast)
    st.markdown("#### Layer 1: Structural DNA (Baseline)")
    fig_struct = go.Figure()
    
    # 1. Raw Price (Translucent)
    fig_struct.add_trace(go.Scatter(x=dates, y=prices, name="Raw Price", line=dict(color="rgba(255,255,255,0.2)")))
    
    # 2. Macro Trend (Structural anchor)
    fig_struct.add_trace(go.Scatter(
        x=dates, y=bands[0], name=band_names[0], 
        line=dict(color="#00ff41", width=3),
        hovertemplate='<b>Structural Trend</b><br>Val: %{y:.2f}<extra></extra>'
    ))
    
    # 3. Spectral Forecast (T+14)
    fig_struct.add_trace(go.Scatter(
        x=future_dates, y=vector, name="Predictive Forecast (T+14)", 
        line=dict(color="#00ff41", dash='dot', width=2),
        hovertemplate='<b>Forecast</b><br>Val: %{y:.2f}<extra></extra>'
    ))
    
    fig_struct.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        font=dict(family="JetBrains Mono"),
        xaxis=dict(gridcolor='#1a1d21', title="Time"),
        yaxis=dict(gridcolor='#1a1d21', title="Price (USD)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_struct, use_container_width=True)

    # 2B. Harmonic Oscillation View (Integrated)
    st.markdown("#### Layer 2: Harmonic Oscillations (Cycles)")
    fig_details = go.Figure()
    hud_palette = ["#ffb300", "#00f0ff", "#ff00ff", "#bc13fe", "#4ade80", "#14b8a6", "#6272a4"]
    
    for i, (band, name) in enumerate(zip(bands[1:], band_names[1:])):
        color = hud_palette[i % len(hud_palette)]
        fig_details.add_trace(go.Scatter(
            x=dates, y=band, name=name,
            line=dict(color=color, width=1.3),
            opacity=0.8,
            hovertemplate=f'<b>{name}</b><br>Val: %{{y:.4f}}<extra></extra>'
        ))
        
    fig_details.update_layout(
        height=500, margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        hovermode="x unified",
        font=dict(family="JetBrains Mono"),
        xaxis=dict(gridcolor='#1a1d21', title="Time"),
        yaxis=dict(gridcolor='#1a1d21', title="Cyclical Volatility"),
        legend=dict(
            orientation="v",
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            itemclick="toggleothers" # High-fidelity isolation
        )
    )
    st.plotly_chart(fig_details, use_container_width=True)

    # 3. Price Vector Detail (Structural Projection)
    st.subheader("Structural Price Vector")
    with st.expander("Vector Methodology Guide", expanded=False):
        st.markdown("""
        The **Price Vector** is a structural projection derived from the first-order derivative of the 
        Approximation Band. It represents the 'momentum vector' of the underlying macro trend, 
        projected 14 days into the future. It is a baseline for structural bias, not a price target.
        """)

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(x=dates, y=prices, name="Actual Price", line=dict(color="rgba(255,255,255,0.4)")))
    fig_proj.add_trace(go.Scatter(x=future_dates, y=vector, name="Projected Structural DNA", 
                                line=dict(color="#00ff41", width=3, dash="dot"),
                                hovertemplate='<b>Projected Structural DNA</b><br>Date: %{x}<br>Val: %{y:.2f}<extra></extra>'))
    
    fig_proj.update_layout(
        title=f"Structural DNA Projection: {symbol}",
        height=400, margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark",
        font=dict(family="JetBrains Mono"),
        xaxis=dict(gridcolor='#1a1d21', title="Time"),
        yaxis=dict(gridcolor='#1a1d21', title="Price (USD)")
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # 4. Cycle Energy Distribution
    st.subheader("Spectral Energy Distribution")
    fig_eng = go.Figure(go.Bar(
        x=band_names, y=band_energy,
        marker_color="#00ff41",
        hovertemplate='<b>Band:</b> %{x}<br><b>Energy Share (Var):</b> %{y:.4f}<extra></extra>'
    ))
    fig_eng.update_layout(
        title="Energy Concentration across Decomposed Layers",
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark",
        font=dict(family="JetBrains Mono"),
        xaxis=dict(gridcolor='#1a1d21'),
        yaxis=dict(gridcolor='#1a1d21')
    )
    st.plotly_chart(fig_eng, use_container_width=True)

else:
    st.error("Select Asset to begin Decomposition Audit.")
