import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.coherence import calculate_coherence, compute_lead_lag_summary
from engine.ui import inject_custom_css

st.set_page_config(page_title="Cross-Asset Resonance | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Cross-Asset Resonance")

# Sidebar
st.sidebar.header("Comparison Settings")
first_sym  = st.sidebar.selectbox("First Asset",  ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=0)
second_sym = st.sidebar.selectbox("Second Asset", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=2)
min_coh_threshold = st.sidebar.slider(
    "Min Coherence for Lead/Lag",
    0.3, 0.9, 0.5, 0.05,
    help="Filter for lead/lag estimates where average coherence exceeds the specified level.",
)

if first_sym == second_sym:
    st.warning("Select two distinct assets for cross-spectral comparison.")
else:
    data1 = get_data(first_sym)
    data2 = get_data(second_sym)

    if data1 is not None and data2 is not None:
        min_size = min(len(data1), len(data2))
        returns1 = calculate_returns(data1).tail(min_size)
        returns2 = calculate_returns(data2).tail(min_size)

        norm1 = z_score_normalize(returns1)
        norm2 = z_score_normalize(returns2)

        st.subheader(f"Cross-Spectral Analysis: {first_sym} vs {second_sym}")
        
        y_scale_type = st.radio("Y-Axis Scale Type", ["Logarithmic (Classic)", "Linear (Structural)"], horizontal=True)

        with st.spinner("Calculating resonance:"):
            sample_size = 750
            dates   = norm1.tail(sample_size).index
            series1 = norm1.tail(sample_size).values
            series2 = norm2.tail(sample_size).values
            resonance_map, phase, coi, freqs, sig = calculate_coherence(series1, series2)

        # Period calculation (1/f)
        # Avoid division by zero
        periods = np.array([1.0 / f if f > 0 else 1000 for f in freqs])
        
        # 1. Coherence Heatmap (Period-Based)
        fig_heat = go.Figure()

        # Add Heatmap
        fig_heat.add_trace(go.Heatmap(
            z=resonance_map,
            x=dates,
            y=periods,
            colorscale="Hot",
            showscale=True,
            colorbar=dict(title="Coherence"),
            zmin=0, zmax=1,
            hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Period:</b> %{y:.1f} Days<br><b>Coherence:</b> %{z:.3f}<extra></extra>'
        ))

        # Add COI Mask (Semi-transparent area)
        # We mask the area WHERE period > coi (or freq < 1/coi)
        # Essentially the 'outside' of the cone.
        coi_periods = coi
        fig_heat.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([coi_periods, [max(periods)] * len(dates)]),
            fill='toself',
            fillcolor='rgba(0, 0, 0, 0.6)',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
            name="Cone of Influence (Unreliable)",
            hoverinfo='skip',
            showlegend=True
        ))

        fig_heat.update_layout(
            title=f"Wavelet Coherence (Time-Period Distribution)",
            xaxis_title="Date",
            yaxis_title="Cycle Period (Days)",
            yaxis_type="log" if "Logarithmic" in y_scale_type else "linear",
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark",
            yaxis=dict(autorange="reversed") # Standard for periods (Fast top, Macro bottom)
        )
        st.plotly_chart(fig_heat, width='stretch')

        # 2. Average Coherence by Period
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            mean_by_freq = np.mean(resonance_map, axis=1)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Scatter(
                x=periods, y=mean_by_freq, fill='tozeroy', mode='lines',
                line=dict(color='#00E676'),
                hovertemplate='<b>Cycle Period:</b> %{x:.1f} Days<br><b>Avg Strength:</b> %{y:.4f}<extra></extra>'
            ))
            fig_bar.update_layout(
                title="Spectral Power Density (by Period)",
                xaxis_title="Cycle Period (Days)",
                yaxis_title="Coherence Strength",
                yaxis_range=[0, 1],
                xaxis_type="log" if "Logarithmic" in y_scale_type else "linear",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_bar, width='stretch')

        with col2:
            st.subheader("Statistical Summary")
            st.markdown(f"""
            #### Structural Terminology
            - **Cycle Period**: The estimated duration of a market rhythm in trading days. 
            - **Nyquist Limit (2.0 Days)**: The fastest discernible period. Corresponds to a frequency of 0.5.
            - **Cone of Influence (COI)**: The shaded region where spectral estimates are contaminated by zero-padding at the signal boundaries.
            
            **Period Mapping:**
            | Period (Days) | Frequency | Classification |
            |---|---|---|
            | 2-5 | 0.5-0.2 | Micro-Volatility |
            | 10-30 | 0.1-0.03 | Swing Momentum |
            | 100+ | < 0.01 | Macro / Structural |

            **Core Statistics**
            | Parameter | Value |
            |---|---|
            | Aggregate Mean Coherence | `{np.mean(resonance_map):.3f}` |
            | Peak Coherence | `{np.max(resonance_map):.3f}` |
            | Primary Resonant Mode | `~{periods[np.argmax(np.mean(resonance_map, axis=1))]:.0f} days` |
            """)

        # 3. Lead / Lag Analysis
        st.divider()
        st.subheader("Phase Angle Lead / Lag Analysis")
        st.markdown(
            "The phase relationship between coherent signals determines directional precedence. "
            "This identifies which asset initiates a move across specific frequency bands."
        )

        summary = compute_lead_lag_summary(
            phase, freqs, resonance_map, coi,
            min_coherence=min_coh_threshold,
        )

        if not summary:
            st.info(
                f"No frequency bands identified with coherence exceeding {min_coh_threshold:.2f} "
                "within the clinical boundary. Adjust the threshold parameters."
            )
        else:
            df = pd.DataFrame(summary)

            colors = [
                "#00c853" if row["first_leads"] else "#ff1744"
                for row in summary
            ]
            fig_ll = go.Figure(go.Bar(
                x=[f"~{r['period_days']:.0f}d cycle" for r in summary],
                y=[r["lead_days"] for r in summary],
                marker_color=colors,
                text=[
                    f"{first_sym} leads by {abs(r['lead_days']):.1f}d"
                    if r["first_leads"]
                    else f"{second_sym} leads by {abs(r['lead_days']):.1f}d"
                    for r in summary
                ],
                textposition="outside",
            ))
            fig_ll.add_hline(y=0, line_color="white", line_width=1)
            fig_ll.update_layout(
                title=f"Lead/Lag Allocation (Green: {first_sym} Leads, Red: {second_sym} Leads)",
                xaxis_title="Cycle Period",
                yaxis_title="Lead/Lag Days",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_ll, width='stretch')

            best     = max(summary, key=lambda r: r["avg_coherence"])
            leader   = first_sym if best["first_leads"] else second_sym
            follower = second_sym if best["first_leads"] else first_sym
            lag_days = abs(best["lead_days"])
            period   = best["period_days"]

            st.success(
                f"Principal Signal: At the **~{period:.0f}-day cycle**, **{leader}** leads **{follower}** "
                f"by **{lag_days:.1f} days** (Coherence: {best['avg_coherence']:.2f})."
            )

            display_df = df.copy()
            display_df["Leader"] = display_df.apply(
                lambda r: f"{first_sym} (+{r['lead_days']:.1f}d)"
                if r["first_leads"] else f"{second_sym} ({r['lead_days']:.1f}d)", axis=1
            )
            display_df = display_df[["period_days", "avg_coherence", "Leader"]].rename(columns={
                "period_days":   "Period (Days)",
                "avg_coherence": "Mean Coherence",
            })
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.caption(
                "Methodological Note: The continuous wavelet transform (CWT) is structurally non-causal. "
                "Phase relationship estimates at localized points in time incorporate adjacent data. "
                "This serves as a structural tendency measurement rather than an execution-ready signal."
            )

    else:
        st.error("Error in data acquisition for the selected asset pair.")
