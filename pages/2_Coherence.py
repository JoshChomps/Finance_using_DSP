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

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Comparison Settings")
first_sym  = st.sidebar.selectbox("First Asset",  ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=0)
second_sym = st.sidebar.selectbox("Second Asset", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=2)
min_coh_threshold = st.sidebar.slider(
    "Min Coherence for Lead/Lag",
    0.3, 0.9, 0.5, 0.05,
    help="Only show lead/lag estimates where average coherence exceeds this level.",
)

if first_sym == second_sym:
    st.warning("Please select two different assets to compare.")
else:
    data1 = get_data(first_sym)
    data2 = get_data(second_sym)

    if data1 is not None and data2 is not None:
        min_size = min(len(data1), len(data2))
        returns1 = calculate_returns(data1).tail(min_size)
        returns2 = calculate_returns(data2).tail(min_size)

        norm1 = z_score_normalize(returns1)
        norm2 = z_score_normalize(returns2)

        st.subheader(f"Analyzing {first_sym} vs {second_sym}")

        with st.spinner("Calculating resonance..."):
            sample_size = 750
            dates   = norm1.tail(sample_size).index
            series1 = norm1.tail(sample_size).values
            series2 = norm2.tail(sample_size).values
            resonance_map, phase, coi, freqs, sig = calculate_coherence(series1, series2)

        # ── 1. Coherence Heatmap ───────────────────────────────────────────────
        fig_heat = go.Figure(data=go.Heatmap(
            z=resonance_map,
            x=dates,
            y=freqs,
            colorscale="Hot",
            showscale=True,
            colorbar=dict(title="Resonance"),
            zmin=0, zmax=1,
            hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Frequency:</b> %{y:.3f}<br><b>Resonance:</b> %{z:.3f}<extra></extra>'
        ))
        fig_heat.add_trace(go.Scatter(
            x=dates,
            y=1.0 / coi,
            name="COI Boundary",
            line=dict(color='white', dash='dash'),
            showlegend=True,
            hoverinfo='skip'
        ))
        fig_heat.update_layout(
            title=f"Wavelet Coherence: {first_sym} vs {second_sym}",
            xaxis_title="Date",
            yaxis_title="Relative Frequency",
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_heat, width='stretch')

        # ── 2. Average Coherence by Frequency ─────────────────────────────────
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            mean_by_freq = np.mean(resonance_map, axis=1)
            periods_days = np.array([1 / f if f > 0.001 else 1000 for f in freqs])

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Scatter(
                x=periods_days, y=mean_by_freq, fill='tozeroy', mode='lines',
                line=dict(color='#00E676'),
                hovertemplate='<b>Cycle Period:</b> %{x:.1f} Days<br><b>Avg Strength:</b> %{y:.4f}<extra></extra>'
            ))
            fig_bar.update_layout(
                title="Average Strength Per Cycle",
                xaxis_title="Period (Days per Cycle)",
                yaxis_title="Strength",
                yaxis_range=[0, 1],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_bar, width='stretch')

        with col2:
            st.subheader("Qualitative Summary")
            st.markdown(f"""
            #### How to read this chart
            - **Glowing Areas**: Strong resonance. {first_sym} and {second_sym} are moving
              together at these specific cycle speeds.
            - **Dark Areas**: Complete decoupling. The assets are charting their own paths.
            - **Dashed V-Shape**: The "Cone of Influence". Ignore data outside this cone,
              as boundary math artifacts can distort the signal.

            **Note on the Y-Axis (max 0.5):**
            The frequency only goes up to `0.5` due to the Nyquist Limit — the fastest
            measurable cycle with daily data takes 2 days (1 cycle / 2 days = 0.5 Hz).
            For higher frequencies, you need intraday data.

            **Quick Stats**
            | | |
            |---|---|
            | Overall avg resonance | `{np.mean(resonance_map):.3f}` |
            | Peak resonance | `{np.max(resonance_map):.3f}` |
            | Most coherent cycle | `~{1/freqs[np.argmax(np.mean(resonance_map, axis=1))]:.0f} days` |
            """)

        # ── 3. Lead / Lag Analysis ────────────────────────────────────────────
        st.divider()
        st.subheader("⏱ Lead / Lag Analysis")
        st.markdown(
            "The **phase angle** between two coherent assets reveals which one moves "
            "first and by how many days — a directional trading edge that standard "
            "correlation metrics cannot provide."
        )

        summary = compute_lead_lag_summary(
            phase, freqs, resonance_map, coi,
            min_coherence=min_coh_threshold,
        )

        if not summary:
            st.info(
                f"No frequency bands found with coherence ≥ {min_coh_threshold:.2f} "
                "inside the cone of influence. Try lowering the threshold."
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
                title=f"Lead/Lag Days by Cycle Period "
                      f"(green = {first_sym} leads, red = {second_sym} leads)",
                xaxis_title="Cycle Period",
                yaxis_title="Lead Days (+ve = first asset leads)",
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
                f"**Strongest signal** (coherence {best['avg_coherence']:.2f}): "
                f"At the **~{period:.0f}-day cycle**, **{leader}** leads **{follower}** "
                f"by approximately **{lag_days:.1f} days**. "
                f"Recent moves in {leader} tend to be followed by similar moves "
                f"in {follower} within that window at this frequency."
            )

            display_df = df.copy()
            display_df["Leader"] = display_df.apply(
                lambda r: f"{first_sym} (+{r['lead_days']:.1f}d)"
                if r["first_leads"] else f"{second_sym} ({r['lead_days']:.1f}d)", axis=1
            )
            display_df = display_df[["period_days", "avg_coherence", "Leader"]].rename(columns={
                "period_days":   "Cycle Period (days)",
                "avg_coherence": "Avg Coherence",
            })
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.caption(
                "⚠️ The CWT is non-causal: phase estimates at time *t* incorporate "
                "nearby future data. Treat this as a structural tendency, not a "
                "precise real-time signal. Validate any strategy out-of-sample."
            )

    else:
        st.error("We had trouble loading data for those specific symbols.")
