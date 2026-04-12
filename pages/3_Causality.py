import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.granger import analyze_causal_flow, check_standard_causality
from engine.intelligence import analyze_causality, get_execution_playbook
from engine.ui import inject_custom_css

st.set_page_config(page_title="Spectral Causality Matrix | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Spectral Causality Matrix")

# == Sidebar ====================================================================
st.sidebar.header("Variable Interaction")
cand_sym   = st.sidebar.selectbox("Predictor Asset (Source)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "BTC-USD"], index=2) 
target_sym = st.sidebar.selectbox("Response Asset (Sink)",    ["SPY", "QQQ", "GLD", "TLT", "AAPL", "MSFT", "BTC-USD"], index=0) 

max_lags = st.sidebar.slider("Causal Lags (p-order)", 1, 25, 5)
x_axis_mode = st.sidebar.radio("X-Axis Resolution", ["Cycle Period (Days)", "Relative Frequency"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Intelligence Decoder")
st.sidebar.markdown("""
**Geweke Spectral Measure**:
Quantifies the reduction in frequency-domain variance of Asset A afforded by the history of Asset B.
- **Intrinsic Power**: Variance that cannot be predicted by external factors.
- **Causal Power**: Strength of Information Transfer.
""")

# == Load and Prep ==============================================================
if cand_sym == target_sym:
    st.error("Identification required: Select distinct source and sink assets.")
else:
    data1 = get_data(cand_sym)
    data2 = get_data(target_sym)
    
    if data1 is not None and data2 is not None:
        min_size = min(len(data1), len(data2))
        rets1 = calculate_returns(data1).tail(min_size)
        rets2 = calculate_returns(data2).tail(min_size)
        
        combined = pd.concat([rets1, rets2], axis=1).dropna()
        combined.columns = [cand_sym, target_sym]
        
        st.subheader(f"Direct Information Transfer: {cand_sym} -> {target_sym}")
        
        with st.spinner("Decomposing causal vectors:"):
            input_data = combined.tail(1000).values
            freq_bins, flow_y_to_x, flow_x_to_y = analyze_causal_flow(input_data, maxlag=max_lags)

        # == 0. Strategy Analysis Matrix ==========================
        st.subheader("Strategy Analysis Matrix")
        
        # Define flow metrics BEFORE the decoder
        flow_delta = np.sum(flow_x_to_y) - np.sum(flow_y_to_x)
        leadership_label = cand_sym if flow_delta > 0.005 else (target_sym if flow_delta < -0.005 else "Neutral (Coupled)")
        
        # Calculate p-values first for the decoder
        p_values = check_standard_causality(combined[[target_sym, cand_sym]].tail(1000).values, lags=max_lags)
        regime, sig_conf, description = analyze_causality(cand_sym, target_sym, flow_delta, np.max(flow_x_to_y), p_values)

        with st.expander("Primary Causal Intelligence", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**SOURCE REGIME**")
                st.header(regime)
                
                st.markdown("**FLOW CONFIDENCE**")
                st.progress(sig_conf)
                st.caption(f"Statistical Significance: {sig_conf:.3f}")
                
            with col2:
                st.markdown("**Analysis Methodology**")
                st.write(f"Information flow analysis identifies {cand_sym} as the **{regime}**. {description}")
                
                # Attributed Lead Horizon
                peak_idx = np.argmax(flow_x_to_y)
                periods = [1.0/f if f > 0 else 1000 for f in freq_bins]
                peak_period = periods[peak_idx]
                st.markdown(f"**Tactical Synchronization**: The strongest predictive flow is concentrated at the **~{peak_period:.1f}d** cycle.")

                # Execution Playbook Injection
                st.markdown("**Execution Playbook**")
                playbook = get_execution_playbook("Causality", regime)
                for step in playbook:
                    st.write(step)

        st.divider()

        # == 1. Integrated Information Flow Dashboard ===========================
        flow_delta = np.sum(flow_x_to_y) - np.sum(flow_y_to_x)
        leadership_label = cand_sym if flow_delta > 0.005 else (target_sym if flow_delta < -0.005 else "Neutral (Coupled)")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Flow Force", f"{np.max(flow_x_to_y):.4f}")
        m2.metric("Leadership Delta", f"{flow_delta:.4f}", delta=f"{leadership_label}")
        m3.metric("Lags Optimized", f"{max_lags}")
        m4.metric("Feedback Sync", "High" if np.max(flow_y_to_x) > 0.05 else "Direct")

        st.divider()

        # Translate frequency to periods for intuitive UI
        periods = [1.0/f if f > 0 else 1000 for f in freq_bins]
        x_data = periods if "Cycle" in x_axis_mode else freq_bins

        fig_gc = go.Figure()
        fig_gc.add_trace(go.Scatter(
            x=x_data, y=flow_x_to_y, name=f"{cand_sym} Lead Intensity", fill='tozeroy',
            line=dict(color='#00ff41', width=2),
            hovertemplate='<b>Cycle:</b> %{x:.1f}d<br><b>Flow:</b> %{y:.4f}<extra></extra>'
        ))
        fig_gc.add_trace(go.Scatter(
            x=x_data, y=flow_y_to_x, name=f"{target_sym} Feedback",
            line=dict(color='#ffb300', width=1, dash='dot'),
            hovertemplate='<b>Cycle:</b> %{x:.1f}d<br><b>Flow:</b> %{y:.4f}<extra></extra>'
        ))
        
        fig_gc.update_layout(
            title="Spectral Information-Flow Topology",
            xaxis_title="Cycle Period (Trading Days)" if "Cycle" in x_axis_mode else "Normalized Frequency",
            yaxis_title="Geweke Causality Measure",
            height=450, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_type="log" if "Cycle" in x_axis_mode else "linear",
            template="plotly_dark",
            font=dict(family="JetBrains Mono"),
            xaxis=dict(gridcolor='#1a1d21'),
            yaxis=dict(gridcolor='#1a1d21'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        if "Cycle" in x_axis_mode:
            fig_gc.update_xaxes(autorange="reversed")
            
        st.plotly_chart(fig_gc, use_container_width=True)
        
        # == 2. Spectral Flow Audit (Causal Tables) =====================
        st.divider()
        col_l, col_r = st.columns([1, 1.2])
        
        with col_l:
            st.subheader("Lag-Order Significance")
            p_stats = pd.DataFrame({"Lag Order": list(p_values.keys()), "p-Value": list(p_values.values())})
            p_stats['Status'] = p_stats['p-Value'].apply(lambda x: "SIGNIFICANT" if x < 0.05 else "STOCHASTIC")
            
            st.table(p_stats.head(8))
            st.caption("Lower p-values (<0.05) indicate highly directed information transfer.")
                
        with col_r:
            st.subheader("Causal Flow Intelligence")
            st.markdown(f"""
            **Audit Profile**:
            The information transfer from **{cand_sym}** to **{target_sym}** is currently classified as **{regime}**. 
            
            **Tactical Implication**:
            - **Leading Node**: {cand_sym if flow_delta > 0 else target_sym} demonstrates structural precedence.
            - **Feedback Loop**: {target_sym if flow_delta > 0 else cand_sym} provides {'verified feedback' if np.max(flow_y_to_x) > 0.02 else 'minimal response'} shocks.
            
            **Execution Advice**:
            Entry/Exit strategies on the response asset should be lagged against source moves using a **{max_lags}-day** lookback window to minimize capture latency.
            """)
            
    else:
        st.error("Acquisition failure: Selected asset series lack sufficient history overlap.")
