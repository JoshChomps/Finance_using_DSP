import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.granger import analyze_causal_flow, check_standard_causality
from engine.ui import inject_custom_css

st.set_page_config(page_title="Directional Causality | FinSignal Suite", layout="wide")
inject_custom_css(st)

st.title("Directional Causality")

# Sidebar Controls
st.sidebar.header("Asset Selection")
candidate = st.sidebar.selectbox("Potential Leader", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=2) 
target_asset = st.sidebar.selectbox("Follower (Target)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=0) 

max_lags = st.sidebar.slider("Analysis Lags", 1, 20, 5)

if candidate == target_asset:
    st.warning("Please choose two different assets.")
else:
    # Load Data
    data1 = get_data(candidate)
    data2 = get_data(target_asset)
    
    if data1 is not None and data2 is not None:
        # Align data lengths
        min_size = min(len(data1), len(data2))
        returns1 = calculate_returns(data1).tail(min_size)
        returns2 = calculate_returns(data2).tail(min_size)
        
        # Merge for leadership check
        combined = pd.concat([returns1, returns2], axis=1).dropna()
        combined.columns = [candidate, target_asset]
        
        st.subheader(f"Causal Path: {candidate} is leading {target_asset}?")
        
        with st.spinner("Analyzing causal flow..."):
            # Use last 1000 points for stable stats
            input_data = combined.tail(1000).values
            # analyze_causal_flow returns (bins, flow_yx, flow_xy):
            #   flow_yx = Y-causes-X = target_asset causes candidate
            #   flow_xy = X-causes-Y = candidate causes target_asset
            freq_bins, flow_target_to_cand, flow_cand_to_target = analyze_causal_flow(input_data, maxlag=max_lags)

        # 1. Spectral Granger Plot
        fig_gc = go.Figure()
        fig_gc.add_trace(go.Scatter(
            x=freq_bins, y=flow_cand_to_target, name=f"{candidate} -> {target_asset}", line=dict(color='orange', width=3),
            hovertemplate='<b>Frequency:</b> %{x:.3f}<br><b>Causal Power:</b> %{y:.4f}<extra></extra>'
        ))
        fig_gc.add_trace(go.Scatter(
            x=freq_bins, y=flow_target_to_cand, name=f"{target_asset} -> {candidate}", line=dict(color='blue', width=3),
            hovertemplate='<b>Frequency:</b> %{x:.3f}<br><b>Causal Power:</b> %{y:.4f}<extra></extra>'
        ))
        
        fig_gc.update_layout(
            title="Frequency-based Leadership Strength",
            xaxis_title="Relative Cycle Frequency",
            yaxis_title="Influence Strength",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_gc, use_container_width=True)
        
        # 2. Time-Domain Table
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Standard Prediction Test")
            with st.spinner("Checking p-values..."):
                # Check if Candidate leads Target
                p_values = check_standard_causality(combined[[target_asset, candidate]].tail(1000).values, lags=max_lags)
                p_stats = pd.DataFrame({"Lag": list(p_values.keys()), "p-value": list(p_values.values())})
                
                def highlight_significant(val):
                    color = 'green' if val < 0.05 else 'None'
                    return f'background-color: {color}'

                # Styler.applymap was renamed to Styler.map in pandas 2.1
                styler = p_stats.style
                apply_fn = getattr(styler, 'map', None) or styler.applymap
                st.table(apply_fn(highlight_significant, subset=['p-value']))
                
        with col2:
            st.subheader("Qualitative Summary")
            st.markdown(f"""
            #### How to read this chart
            - **Influence Strength**: Peaks in the graph show the cycles where information flows strongest from one asset to another.
            - **Frequency Spectrum**:
                - High frequency (>0.3): Rapid noise or sentiment-driven leadership.
                - Low frequency (<0.1): Long-term structural or macro leadership.
            
            **Note on the X-Axis (max 0.5):**
            The cycle frequency caps at `0.5`, representing the mathematics of the "Nyquist limit". Because we are feeding the engine 1 data point per day, the absolute fastest oscillation we can measure takes exactly 2 days to cycle (1 cycle / 2 days = 0.5).

            **Why Spectral Causality?**
            Standard models just ask: *Does {candidate} move before {target_asset}?* 
            Spectral Granger Causality asks a better question: *Does {candidate} lead {target_asset} during weekly swings, or does it only lead during massive macro trends?* This allows you to build frequency-specific trading pairs.
            
            **Observation:**
            The strongest leadership from **{candidate}** to **{target_asset}** occurs at the `{freq_bins[np.argmax(flow_cand_to_target)]:.3f}` frequency. 
            """)
            
    else:
        st.error("There was an issue loading the asset data.")

