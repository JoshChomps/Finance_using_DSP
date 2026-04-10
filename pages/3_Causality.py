import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import compute_log_returns, normalize_signal
from engine.granger import compute_spectral_granger, compute_time_domain_granger

st.set_page_config(page_title="Directional Causality | FinSignal Suite", layout="wide")

st.title("➡️ Directional Causality")

# Sidebar Controls
st.sidebar.header("Settings")
t1 = st.sidebar.selectbox("Leading Asset (Candidate)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=2) # Default GLD leads
t2 = st.sidebar.selectbox("Following Asset (Target)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT"], index=0) # Default SPY

maxlag = st.sidebar.slider("VAR Max Lag", 1, 20, 5)

if t1 == t2:
    st.warning("Please select two different tickers.")
else:
    # Load Data
    d1 = get_data(t1)
    d2 = get_data(t2)
    
    if d1 is not None and d2 is not None:
        # Align data
        min_len = min(len(d1), len(d2))
        r1 = compute_log_returns(d1).tail(min_len)
        r2 = compute_log_returns(d2).tail(min_len)
        
        # Combine into a single dataframe for VAR
        # column 0 is target (x), column 1 is candidate leader (y)
        # We want to see if y leads x
        df_var = pd.concat([r1, r2], axis=1).dropna()
        df_var.columns = [t1, t2]
        
        st.subheader(f"Causal Path: {t1} ↔ {t2}")
        
        with st.spinner("Computing Spectral Granger Causality..."):
            # Using only last 1000 points for stability
            data_arr = df_var.tail(1000).values
            freqs, g_yx, g_xy = compute_spectral_granger(data_arr, maxlag=maxlag)
            
        # 1. Spectral Granger Plot
        fig_gc = go.Figure()
        fig_gc.add_trace(go.Scatter(x=freqs, y=g_yx, name=f"{t1} -> {t2}", line=dict(color='orange', width=3)))
        fig_gc.add_trace(go.Scatter(x=freqs, y=g_xy, name=f"{t2} -> {t1}", line=dict(color='blue', width=3)))
        
        fig_gc.update_layout(
            title="Spectral Granger Causality (Geweke Measure)",
            xaxis_title="Normalized Frequency (cycles/day)",
            yaxis_title="Causal Strength",
            height=500
        )
        st.plotly_chart(fig_gc, use_container_width=True)
        
        # 2. Time-Domain Comparison
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Time-Domain Analysis")
            with st.spinner("Computing Time-Domain Granger..."):
                # Check if T1 leads T2
                p_vals = compute_time_domain_granger(df_var[[t2, t1]].tail(1000).values, maxlag=maxlag)
                p_df = pd.DataFrame({"Lag": list(p_vals.keys()), "p-value": list(p_vals.values())})
                
                def highlight_sig(val):
                    color = 'green' if val < 0.05 else 'white'
                    return f'background-color: {color}'
                
                st.table(p_df.style.applymap(highlight_sig, subset=['p-value']))
                
        with col2:
            st.subheader("Interpretation")
            st.markdown(f"""
            - **Causal Strength**: Higher values indicate stronger leadership at that specific frequency.
            - **Frequency Scaling**:
                - High frequency (>0.3) = Intraday/Noise coupling.
                - Low frequency (<0.1) = Macro/Structural leadership.
            
            **Current Context:**
            At the `{freqs[np.argmax(g_yx)]:.3f}` frequency, **{t1}** has its strongest causal influence on **{t2}**.
            """)
            
    else:
        st.error("Error loading data.")
