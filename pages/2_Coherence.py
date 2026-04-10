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

st.title("🤝 Cross-Asset Resonance")

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
            series1 = norm1.tail(sample_size).values
            series2 = norm2.tail(sample_size).values
            
            resonance_map, phase, coi, freqs, sig = calculate_coherence(series1, series2)
            
        # 1. Coherence Heatmap
        fig_heat = go.Figure(data=go.Heatmap(
            z=resonance_map,
            x=np.arange(len(series1)),
            y=freqs,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="Resonance")
        ))
        
        # Dash line for Cone of Influence
        fig_heat.add_trace(go.Scatter(
            x=np.arange(len(series1)),
            y=1.0/coi, 
            name="Boundary Artifacts",
            line=dict(color='white', dash='dash'),
            showlegend=False
        ))

        fig_heat.update_layout(
            title=f"Wavelet Coherence: {first_sym} vs {second_sym}",
            xaxis_title="Time index (Days)",
            yaxis_title="Relative Frequency",
            height=600
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # 2. Insights & Summary
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            mean_by_freq = np.mean(resonance_map, axis=1)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=freqs, y=mean_by_freq))
            fig_bar.update_layout(title="Average Strength Per Cycle", xaxis_title="Frequency", yaxis_title="Strength")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.markdown(f"""
            #### How to read this chart
            - **Glowing Areas**: Strong resonance. These assets are moving together at these specific cycles.
            - **Dark Areas**: No connection. The assets are decoupling.
            - **Dashed V-Shape**: This marks the limit where edge effects might skew the math.
            
            **Quick Stats:**
            - Average Resonance: `{np.mean(resonance_map):.4f}`
            - Peak Resilience: `{np.max(resonance_map):.4f}`
            """)
            
    else:
        st.error("We had trouble loading data for those specific symbols.")

