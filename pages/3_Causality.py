import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.granger import analyze_causal_flow, check_standard_causality
from engine.ui import inject_custom_css

st.set_page_config(page_title="Directional Causality | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Directional Causality")

# Sidebar
st.sidebar.header("Asset Selection")
candidate     = st.sidebar.selectbox("Predictor Asset (Candidate)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=2) 
target_asset  = st.sidebar.selectbox("Response Asset (Target)",    ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=0) 

max_lags = st.sidebar.slider("Analysis Lags", 1, 20, 5)

if candidate == target_asset:
    st.warning("Selection required: two distinct assets for causality analysis.")
else:
    data1 = get_data(candidate)
    data2 = get_data(target_asset)
    
    if data1 is not None and data2 is not None:
        min_size = min(len(data1), len(data2))
        returns1 = calculate_returns(data1).tail(min_size)
        returns2 = calculate_returns(data2).tail(min_size)
        
        combined = pd.concat([returns1, returns2], axis=1).dropna()
        combined.columns = [candidate, target_asset]
        
        st.subheader(f"Causal Path Investigation: {candidate} -> {target_asset}")
        
        with st.spinner("Analyzing causal flow:"):
            input_data = combined.tail(1000).values
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
            title="Frequency-Domain Causal Strength",
            xaxis_title="Relative Frequency",
            yaxis_title="Information Flow (Influence Strength)",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_gc, width='stretch')
        
        # 2. Time-Domain Validation
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Standard Predictive Significance")
            with st.spinner("Calculating p-values:"):
                p_values = check_standard_causality(combined[[target_asset, candidate]].tail(1000).values, lags=max_lags)
                p_stats = pd.DataFrame({"Lag": list(p_values.keys()), "p-value": list(p_values.values())})
                
                def highlight_significant(val):
                    if val < 0.05:
                        return 'background-color: rgba(46, 160, 67, 0.2)'
                    return ''

                styler = p_stats.style
                apply_fn = getattr(styler, 'map', None) or styler.applymap
                st.table(apply_fn(highlight_significant, subset=['p-value']))
                
        with col2:
            st.subheader("Statistical Interpretation")
            st.markdown(f"""
            #### Methodology Note
            - **Information Flow**: Spectral peaks identify cycle frequencies where predictive leadership is statistically significant.
            - **Frequency Distribution**:
                - High frequency (>0.3): Dominance of short-term noise or sentiment-driven precedence.
                - Low frequency (<0.1): Structural macro-economic leadership.
            
            **Nyquist Resolution:**
            Measurements are constrained to a maximum frequency of 0.5 cycles per unit.

            **Spectral Granger Causality Benefits:**
            As opposed to aggregate time-domain causality, the frequency-domain approach decomposes leadership into specific market rhythms. This allows for the identification of lead-lag relationships that may exist during long-term trends but vanish during short-term volatility.
            
            **Observation Summary:**
            """)
            max_flow = float(np.max(flow_cand_to_target)) if len(flow_cand_to_target) > 0 else 0.0
            if max_flow > 0.01:
                peak_freq = freq_bins[np.argmax(flow_cand_to_target)]
                st.markdown(f"The statistical peak for causal flow from **{candidate}** to **{target_asset}** is observed at the `{peak_freq:.3f}` frequency bin.")
            else:
                st.markdown(f"No significant frequency-domain leadership detected from **{candidate}** to **{target_asset}** in this sample.")
            
    else:
        st.error("Error in data retrieval for the selected assets.")

