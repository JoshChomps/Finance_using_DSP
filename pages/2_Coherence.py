import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.coherence import calculate_coherence, compute_lead_lag_summary
from engine.ui import inject_custom_css

st.set_page_config(page_title="Cross-Asset Resonance | FinSignal Suite", layout="wide")
inject_custom_css(st)

st.title("🤝 Cross-Asset Resonance")

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

        # Convert frequencies to periods (days) for a more intuitive y-axis
        periods = np.array([1.0 / f if f > 0 else np.nan for f in freqs])

        # ── 1. Coherence Heatmap ───────────────────────────────────────────────
        fig_heat = go.Figure(data=go.Heatmap(
            z=resonance_map,
            x=dates,
            y=periods,
            colorscale="Hot",
            showscale=True,
            colorbar=dict(title="Coherence"),
            zmin=0, zmax=1,
            hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Period:</b> %{y:.1f} days<br><b>Coherence:</b> %{z:.3f}<extra></extra>',
        ))
        # COI overlay — shade the unreliable outer region
        fig_heat.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([coi, [np.nanmax(periods)] * len(dates)]),
            fill='toself',
            fillcolor='rgba(0, 0, 0, 0.55)',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
            name="Outside COI (unreliable)",
            hoverinfo='skip',
        ))
        fig_heat.update_layout(
            title=f"Wavelet Coherence: {first_sym} vs {second_sym}",
            xaxis_title="Date",
            yaxis_title="Cycle Period (days)",
            yaxis=dict(autorange="reversed"),   # fast cycles at top, macro at bottom
            height=550,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── 2. Average Coherence by Frequency ─────────────────────────────────
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            mean_by_freq = np.mean(resonance_map, axis=1)
            period_labels = [f"{1/f:.0f}d" if f > 0 else "∞" for f in freqs]
            fig_bar = go.Figure(go.Bar(
                x=period_labels, y=mean_by_freq,
                marker_color=mean_by_freq,
                marker_colorscale="Hot",
            ))
            fig_bar.update_layout(
                title="Average Coherence by Cycle Period",
                xaxis_title="Cycle Period",
                yaxis_title="Mean Coherence",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown(f"""
            #### How to read this
            - **Bright/hot areas** on the heatmap: these two assets are strongly
              synchronized at that frequency and time window.
            - **Dark areas**: the assets are decoupled — moving independently.
            - **Dashed boundary**: the Cone of Influence. Results outside it are
              unreliable due to edge effects in the wavelet transform.

            **Quick Stats**
            | | |
            |---|---|
            | Overall avg coherence | `{np.mean(resonance_map):.3f}` |
            | Peak coherence | `{np.max(resonance_map):.3f}` |
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

            # Build a clear lead/lag bar chart (positive = first leads, negative = second leads)
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
                title=f"Lead/Lag Days by Cycle Period  "
                      f"(green = {first_sym} leads, red = {second_sym} leads)",
                xaxis_title="Cycle Period",
                yaxis_title="Lead Days (+ve = first asset leads)",
                height=400,
            )
            st.plotly_chart(fig_ll, use_container_width=True)

            # Actionable insight callout
            best = max(summary, key=lambda r: r["avg_coherence"])
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

            # Summary table
            display_df = df.copy()
            display_df["Leader"] = display_df.apply(
                lambda r: f"{first_sym} (+{r['lead_days']:.1f}d)"
                if r["first_leads"] else f"{second_sym} ({r['lead_days']:.1f}d)", axis=1
            )
            display_df = display_df[["period_days", "avg_coherence", "Leader"]].rename(columns={
                "period_days":    "Cycle Period (days)",
                "avg_coherence":  "Avg Coherence",
            })
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.caption(
                "⚠️ The CWT is non-causal: phase estimates at time *t* incorporate "
                "nearby future data. Treat this as a structural tendency, not a "
                "precise real-time signal. Validate any strategy out-of-sample."
            )

    else:
        st.error("We had trouble loading data for those specific symbols.")
