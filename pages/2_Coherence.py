import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.coherence import calculate_coherence
from engine.ui import inject_custom_css

st.set_page_config(page_title="Cross-Asset Resonance | FinSignal Suite", layout="wide")
inject_custom_css(st)

st.title("Cross-Asset Resonance")

# Sidebar Controls
st.sidebar.header("Comparison Settings")
first_sym = st.sidebar.selectbox("First Asset", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=0)
second_sym = st.sidebar.selectbox("Second Asset", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=2) 

if first_sym == second_sym:
    st.warning("Please select two different assets to compare.")
else:
    # Load Data
    data1 = get_data(first_sym)
    data2 = get_data(second_sym)
    
    if data1 is not None and data2 is not None:
        # Align data lengths
        min_size = min(len(data1), len(data2))
        returns1 = calculate_returns(data1).tail(min_size)
        returns2 = calculate_returns(data2).tail(min_size)
        
        # Normalize for comparison
        norm1 = z_score_normalize(returns1)
        norm2 = z_score_normalize(returns2)
        
        st.subheader(f"Analyzing {first_sym} vs {second_sym}")
        
        with st.spinner("Calculating resonance..."):
            # Limit to 750 points for smooth dashboard interaction
            sample_size = 750
            dates = norm1.tail(sample_size).index
            series1 = norm1.tail(sample_size).values
            series2 = norm2.tail(sample_size).values
            
            resonance_map, phase, coi, freqs, sig = calculate_coherence(series1, series2)
            
        fig_heat = go.Figure(data=go.Heatmap(
            z=resonance_map,
            x=dates,
            y=freqs,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="Resonance"),
            hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Frequency:</b> %{y:.3f}<br><b>Resonance:</b> %{z:.3f}<extra></extra>'
        ))
        
        # Dash line for Cone of Influence
        fig_heat.add_trace(go.Scatter(
            x=dates,
            y=1.0/coi, 
            name="Boundary Artifacts",
            line=dict(color='white', dash='dash'),
            showlegend=False,
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
        
        # 2. Insights & Summary
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            mean_by_freq = np.mean(resonance_map, axis=1)
            periods_days = np.array([1 / f if f > 0.001 else 1000 for f in freqs])
            
            fig_bar = go.Figure()
            # Frequencies are better plotted with Scatter+fill rather than Bar because they are continuous non-linear
            fig_bar.add_trace(go.Scatter(
                x=periods_days, y=mean_by_freq, fill='tozeroy', mode='lines', line=dict(color='#00E676'),
                hovertemplate='<b>Cycle Period:</b> %{x:.1f} Days<br><b>Avg Strength:</b> %{y:.4f}<extra></extra>'
            ))
            fig_bar.update_layout(
                title="Average Strength Per Cycle", 
                xaxis_title="Period (Days per Cycle)", 
                yaxis_title="Strength",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_bar, width='stretch')
            
        with col2:
            st.subheader("Qualitative Summary")
            st.markdown(f"""
            #### How to read this chart
            - **Glowing Areas**: Strong resonance. {first_sym} and {second_sym} are moving together at these specific cycle speeds.
            - **Dark Areas**: Complete decoupling. The assets are charting their own separate paths.
            - **Dashed V-Shape**: The "Cone of Influence". Ignore data outside this cone, as boundary math artifacts can distort the signal.
            
            **Note on the Y-Axis (max 0.5):**
            You might notice the frequency only goes up to `0.5`. This isn't amplitude, it's measuring cycles per day. Due to the "Nyquist Limit" of daily data, the fastest cycle we can mathematically measure takes 2 days to complete (1 cycle / 2 days = 0.5 frequency). To see higher frequencies, you would need intraday data (like the live flow module).

            **Why this matters:**
            Standard correlation might tell you these assets are 80% correlated over 5 years. But **Wavelet Coherence** tells you *exactly when* and *at what speed*. For example, they might be highly correlated on a macro scale (low frequency) but completely uncoupled on a daily basis (fast noise).
            
            **Quick Stats:**
            - Average Resonance: `{np.mean(resonance_map):.4f}`
            - Peak Resilience: `{np.max(resonance_map):.4f}`
            """)
            
    else:
        st.error("We had trouble loading data for those specific symbols.")

