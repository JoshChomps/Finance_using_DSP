import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.coherence import calculate_coherence, compute_lead_lag_summary
from engine.intelligence import analyze_resonance, get_execution_playbook
from engine.ui import inject_custom_css

st.set_page_config(page_title="Portfolio Resonance Guardian | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Portfolio Resonance Guardian")

# == Sidebar ====================================================================
st.sidebar.header("Asset Comparison")
first_sym  = st.sidebar.selectbox("Primary Asset", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD"], index=0)
second_sym = st.sidebar.selectbox("Compare Against", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "BTC-USD"], index=1)

st.sidebar.divider()
st.sidebar.header("Analysis Parameters")
analysis_window = st.sidebar.slider("Analysis Window (Days)", 250, 2000, 750, 
                                     help="The lookback period for wavelet coherence calculation.")
y_scale_type = st.sidebar.radio("Spectral Resolution", ["Logarithmic (Classic)", "Linear (Structural)"], index=0)

# == Load and Prep ==============================================================
if first_sym == second_sym:
    st.error("Select distinct assets for resonance analysis.")
else:
    data1 = get_data(first_sym)
    data2 = get_data(second_sym)

    if data1 is not None and data2 is not None:
        returns1 = calculate_returns(data1)
        returns2 = calculate_returns(data2)

        # Sync dates
        common_idx = returns1.index.intersection(returns2.index)
        if len(common_idx) > 200:
            norm1 = z_score_normalize(returns1.loc[common_idx])
            norm2 = z_score_normalize(returns2.loc[common_idx])
            with st.spinner("Calculating lead/lag resonance..."):
                dates   = norm1.tail(analysis_window).index
                series1 = norm1.tail(analysis_window).values
                series2 = norm2.tail(analysis_window).values
                resonance_map, phase_map, coi, freqs, sig = calculate_coherence(series1, series2)
                
                # Lead/Lag Attribution
                summary = compute_lead_lag_summary(phase_map, freqs, resonance_map, coi)
                regime, avg_coh, description = analyze_resonance(summary)

            periods = np.array([1.0 / f if f > 0 else 1000 for f in freqs])
            
            # == 0. Strategy Analysis Matrix ==========================
            st.subheader("Strategy Analysis Matrix")
            with st.expander("Primary Resonance Intelligence", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**CURRENT REGIME**")
                    st.header(regime)
                    
                    st.markdown("**RESONANCE FORCE**")
                    st.progress(avg_coh)
                    st.caption(f"Weighted Spectral Alignment: {avg_coh:.3f}")
                    
                with col2:
                    st.markdown("**Analysis Methodology**")
                    st.write(f"The engine detects a **{regime}** state between {first_sym} and {second_sym}. {description}")
                    
                    if summary:
                        dominant = summary[np.argmax([r['avg_coherence'] for r in summary])]
                        lead_text = f"leads (by {dominant['lead_days']}d)" if dominant['lead_days'] > 0 else f"lags (by {abs(dominant['lead_days'])}d)"
                        st.markdown(f"**Tactical Lead/Lag Attribution**: The primary asset Currently **{lead_text}** on the dominant **~{dominant['period_days']}d** cycle.")

                    # Execution Playbook Injection
                    st.markdown("**Execution Playbook**")
                    playbook = get_execution_playbook("Coherence", regime)
                    for step in playbook:
                        st.write(step)

            # == 1. Coherence Heatmap (Wavelet Intensity) =======================
            st.subheader(f"Cross-Spectral Coherence: {first_sym} vs {second_sym}")
            fig_heat = go.Figure()

            fig_heat.add_trace(go.Heatmap(
                z=resonance_map,
                x=dates,
                y=periods,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Resonance", thickness=15),
                zmin=0, zmax=1,
                hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Period:</b> %{y:.1f}d<br><b>Strength:</b> %{z:.3f}<extra></extra>'
            ))

            # Add COI Mask
            fig_heat.add_trace(go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([coi, [max(periods)] * len(dates)]),
                fill='toself', fillcolor='rgba(0, 0, 0, 0.6)',
                line=dict(color='rgba(255, 255, 255, 0.1)', width=0.5),
                name="Cone of Influence", hoverinfo='skip', showlegend=True
            ))

            # Dynamic Y-Axis Ticks for Log Scale Clarity
            tick_vals = [2, 5, 10, 20, 40, 60, 120, 252, 500]
            tick_text = [str(v) for v in tick_vals]

            fig_heat.update_layout(
                height=450, margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_dark",
                font=dict(family="JetBrains Mono"),
                yaxis_title="Cycle Period (Days)",
                yaxis_type="log" if "Logarithmic" in y_scale_type else "linear",
                yaxis=dict(
                    autorange="reversed", 
                    gridcolor='#1a1d21',
                    tickvals=tick_vals if "Logarithmic" in y_scale_type else None,
                    ticktext=tick_text if "Logarithmic" in y_scale_type else None
                ),
                xaxis=dict(gridcolor='#1a1d21')
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # == 2. Cross-Spectral Phase (Lead/Lag Topology) ====================
            st.subheader("Cross-Spectral Phase Topology")
            with st.expander("Interpretation Guide: Phase Angle", expanded=False):
                st.markdown("""
                **Phase Angle (Radians)** indicates the temporal shift between two assets:
                - **0 rad**: Perfect in-phase synchronization.
                - **+pi/2 rad**: Primary asset leads by 1/4 cycle.
                - **-pi/2 rad**: Primary asset lags by 1/4 cycle.
                - **pi rad**: Perfect anti-phase (inverse) correlation.
                """)

            target_idx = st.selectbox("Isolate Frequency Band for Phase Analysis", range(len(freqs)),
                                      format_func=lambda i: f"~{periods[i]:.0f}-day cycle", index=len(freqs)//3)
            
            fig_pha = go.Figure()
            fig_pha.add_trace(go.Scatter(x=dates, y=phase_map[target_idx], name="Phase Angle", line=dict(color='#a78bfa', width=2)))
            fig_pha.add_hline(y=0, line_dash="dash", line_color="white")
            fig_pha.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_dark",
                yaxis_title="Phase (Radians)",
                yaxis_range=[-np.pi, np.pi]
            )
            st.plotly_chart(fig_pha, use_container_width=True)

            # == 3. Lead-Lag Audit Table (Detailed Attribution) =================
            st.divider()
            st.subheader("Cross-Spectral Signal Audit")
            if summary:
                audit_df = pd.DataFrame(summary)
                # Formatting for the HUD
                audit_df['Status'] = audit_df['lead_days'].apply(lambda x: "LEADING" if x > 0 else "LAGGING")
                audit_df = audit_df[['period_days', 'avg_coherence', 'lead_days', 'Status']]
                audit_df.columns = ['Period (Days)', 'Resonance Strength', 'Lead/Lag (Days)', 'Structural Status']
                
                st.table(audit_df.sort_values('Resonance Strength', ascending=False).head(8))
                st.caption(f"Audit Verification: Lead/Lag values represent a {first_sym} vs {second_sym} temporal offset. Positive = {first_sym} leads.")
            else:
                st.info("Insufficient spectral density for granular cycle attribution.")

            # == 4. Diagnostic Summary ==========================================
            st.divider()
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Spectral Power Distribution")
                mean_res = np.mean(resonance_map, axis=1)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Scatter(x=periods, y=mean_res, fill='tozeroy', line=dict(color='#00ff41')))
                fig_bar.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=30),
                    template="plotly_dark",
                    font=dict(family="JetBrains Mono"),
                    xaxis_title="Cycle Period (Days)",
                    yaxis_title="Avg Coherence",
                    xaxis_type="log" if "Logarithmic" in y_scale_type else "linear",
                    xaxis=dict(gridcolor='#1a1d21'),
                    yaxis=dict(gridcolor='#1a1d21')
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with c2:
                st.subheader("Statistical Inventory")
                st.markdown(f"""
                | Metric | Value |
                |---|---|
                | Aggregate Resonance | `{np.mean(resonance_map):.3f}` |
                | Peak Resonant Node | `{np.max(resonance_map):.3f}` |
                | Dominant Rhythm | `~{periods[np.argmax(mean_res)]:.1f} days` |
                | Window Integrity | `{analysis_window} bars` |
                | Nyquist Guard | `2.0 days` |
                """)
                
                if np.mean(resonance_map) > 0.6:
                    st.success("Target pair demonstrates Institutional Grade resonance stability.")
                else:
                    st.warning("Low resonance detected. Predictive lead/lag signals may be speculative.")

        else:
            st.error("Insufficient historical overlap to conduct resonance analysis.")
    else:
        st.error("Failed to acquire data for selected asset pair.")
