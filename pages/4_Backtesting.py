import streamlit as st # [RELOAD_TRIGGER_V30]
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.coherence import calculate_coherence
from engine.backtest import (
    run_backtest,
    create_signals_from_resonance,
    create_phase_signals,
    compute_kelly_fraction,
    apply_trend_filter,
    coherence_stability,
)
from engine.intelligence import analyze_backtest, get_execution_playbook
from engine.ui import inject_custom_css

st.set_page_config(page_title="Backtesting Simulator | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Backtesting Simulator")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Asset Selection")
traded_asset  = st.sidebar.selectbox("Traded Asset", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=0)
signal_source = st.sidebar.selectbox("Signal Source (Lead/Lag)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=2)

st.sidebar.divider()
st.sidebar.header("Strategy Configuration")
signal_mode = st.sidebar.radio(
    "Signal Generation",
    ["Phase-Following (Clinical)", "Coherence Mean-Reversion"],
    help="Phase-Following uses the wavelet lead/lag offset to determine direction. Mean-Reversion bets on decoupling."
)

if signal_mode == "Phase-Following (Clinical)":
    coh_threshold    = st.sidebar.slider("Resonance Threshold", 0.4, 0.9, 0.6)
    phase_strength   = st.sidebar.slider("Min Phase Lead (rad)", 0.05, 1.0, 0.15)
    phase_smoothing  = st.sidebar.slider("Phase Smoothing (days)", 3, 20, 5)
else:
    high_limit = st.sidebar.slider("Upper Barrier (Short)", 0.5, 0.99, 0.7)
    low_limit  = st.sidebar.slider("Lower Barrier (Long)", 0.01, 0.5, 0.3)

st.sidebar.header("Execution & Risk")
fee_perc = st.sidebar.number_input("Transaction Fee (%)", 0.0, 1.0, 0.05, 0.01) / 100.0

sizing_mode = st.sidebar.radio("Position Sizing", ["Auto - Kelly Fraction", "Fixed Capital"], index=0)
manual_size = st.sidebar.slider("Capital Fraction (%)", 10, 100, 100) / 100.0 if "Fixed" in sizing_mode else None

st.sidebar.header("Signal Conditioning")
use_trend_filter = st.sidebar.checkbox("Structural MA Filter", value=True)
ma_period = st.sidebar.slider("MA Window (days)", 20, 200, 50) if use_trend_filter else None

st.sidebar.header("Validation")
oos_pct = st.sidebar.slider("OOS Test Split (%)", 10, 50, 30, 
                            help="Reserve the end of the data as a held-out test set to detect model overfitting.")

st.sidebar.divider()
st.sidebar.subheader("Intelligence Decoder")
st.sidebar.markdown("""
**Risk Attribution Logic**:
- **T+1 Execution**: Ensures signals generated at Close(t) are executed at Open(t+1) to prevent look-ahead bias.
- **Sortino Ratio**: Standard Deviation of *downside* variance only.
- **Kelly Fraction**: Optimal capital allocation based on the reward-to-risk (mu/sigma²) manifold.
""")

# ── Load and Prep ──────────────────────────────────────────────────────────────
if traded_asset == signal_source:
    st.error("Select distinct assets for lead/lag analysis.")
else:
    data1 = get_data(traded_asset)
    data2 = get_data(signal_source)

    if data1 is not None and data2 is not None:
        common_idx = data1.index.intersection(data2.index)
        rets1 = calculate_returns(data1.loc[common_idx])
        rets2 = calculate_returns(data2.loc[common_idx])
        
        sample_size = 1000
        dates   = rets1.tail(sample_size).index
        z1      = z_score_normalize(rets1.tail(sample_size)).values
        z2      = z_score_normalize(rets2.tail(sample_size)).values
        actual_rets = rets1.tail(sample_size).values
        prices      = data1["Close"].loc[dates].values

        with st.spinner("Calculating lead/lag resonance..."):
            res_grid, pha_grid, coi, freqs, sig = calculate_coherence(z1, z2)
            
        # Phase 21: Auto-Tune Logic
        # Phase 33: Active Band Mask (Exclude Trend/DC)
        avg_coherence = np.nanmean(res_grid, axis=1)
        periods_days = [1/f if f > 0 else 10000 for f in freqs]
        active_mask = (np.array(periods_days) > 5) & (np.array(periods_days) < 252)
        
        if np.any(active_mask):
            masked_coh = avg_coherence.copy()
            masked_coh[~active_mask] = -1
            best_band = int(np.nanargmax(masked_coh))
        else:
            best_band = int(np.nanargmax(avg_coherence))
        
        periods_days = [f"~{1/f:.0f}-day cycle" if f > 0 else "DC" for f in freqs]
        target_idx   = st.selectbox(f"Signal Identification Band (Auto-Tune: {periods_days[best_band]})", 
                                    range(len(freqs)), 
                                    format_func=lambda i: periods_days[i], 
                                    index=best_band)
        
        # Coherence Stability Metrics
        stab = coherence_stability(res_grid, target_idx)
        recent_stab = np.nanmean(stab[-50:]) if len(stab) > 50 else 0.5
        stab_rating = "Institutional Grade" if recent_stab < 0.15 else "Speculative"

        # Build Signal
        if "Phase" in signal_mode:
            signals = create_phase_signals(res_grid, pha_grid, z2, target_idx,
                                           coherence_threshold=coh_threshold,
                                           min_phase_strength=phase_strength,
                                           smoothing=phase_smoothing)
        else:
            signals = create_signals_from_resonance(res_grid, target_idx, high_barrier=high_limit, low_barrier=low_limit)

        if use_trend_filter:
            signals = apply_trend_filter(signals, prices, ma_period=ma_period)

        # Position Sizing
        pos_size = compute_kelly_fraction(actual_rets, signals) if "Kelly" in sizing_mode else manual_size
        
        # ── Execution Logic ────────────────────────────────────────────────────
        oos_start = int(len(actual_rets) * (1 - oos_pct / 100))
        
        # Results calculation
        results_full = run_backtest(actual_rets, signals, slippage=fee_perc, position_size=pos_size)
        results_oos  = run_backtest(actual_rets[oos_start:], signals[oos_start:], slippage=fee_perc, position_size=pos_size)
        
        # Actionable Performance Intelligence
        regime, val_score, description = analyze_backtest(results_full, results_oos)

        # ── 0. Strategy Analysis Matrix ──────────────────────────
        st.subheader("Strategy Analysis Matrix")
        with st.expander("Primary Strategy Intelligence", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**STRATEGY STATUS**")
                st.header(regime)
                
                st.markdown("**VALIDATION STABILITY**")
                st.progress(max(0.01, (val_score + 1) / 2)) # Center at 0.5
                st.caption(f"IS/OOS Sharpe Preservation: {val_score:.2f}")
                
            with col2:
                st.markdown("**Analysis Methodology**")
                st.write(f"The simulation identifies a **{regime}** condition. {description}")
                
                # Execution Playbook Injection
                st.markdown("**Execution Playbook**")
                playbook = get_execution_playbook("Backtesting", regime)
                for step in playbook:
                    st.write(step)
                
                if val_score < 0.2 and results_full['sharpe'] > 0.5:
                    st.warning("Spectral Decay Alert: High in-sample alpha is not translating to out-of-sample stability.")
                elif results_full['total_trades'] < 5:
                    st.info("Sample Size Warning: Strategy expectancy is speculative due to low transaction density.")

        # ── 1. Tactical Strategy Audit ──────────────────────────────────────
        st.divider()
        st.subheader("Strategy Audit: Lead/Lag Validation")
        
        audit_data = {
            "Metric": ["Sharpe Ratio", "Profit Factor", "Win Rate", "Max Drawdown", "Expectancy"],
            "Full Period (IS+OOS)": [
                f"{results_full['sharpe']:.2f}",
                f"{results_full['profit_factor']:.2f}",
                f"{results_full['win_rate']:.1%}",
                f"{results_full['max_drawdown']:.1%}",
                f"{results_full['expectancy']:.4f}"
            ],
            "Post-Anchor (OOS Only)": [
                f"{results_oos['sharpe']:.2f}",
                f"{results_oos['profit_factor']:.2f}",
                f"{results_oos['win_rate']:.1%}",
                f"{results_oos['max_drawdown']:.1%}",
                f"{results_oos['expectancy']:.4f}"
            ]
        }
        st.table(pd.DataFrame(audit_data))
        st.caption(f"Clinical Audit: OOS Anchor point established at bar {oos_start} ({oos_pct}% held-out).")

        # ── 2. Integrated Performance Dashboard ───────────────────────────────
        st.divider()
        st.subheader("Integrated Performance Dashboard")
        
        # Row 1: Standard Attribution
        st.markdown("#### Institutional Attribution (Full Period)")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Annual Return", f"{results_full['annual_return']:.1%}", f"{results_full['final_profit']:.1%} Total")
        f2.metric("Sharpe Ratio", f"{results_full['sharpe']:.2f}", stab_rating)
        f3.metric("Sortino Ratio", f"{results_full['sortino']:.2f}", help="Standardized Downside Deviation (RMS of negative returns).")
        f4.metric("Calmar Ratio", f"{results_full['calmar']:.2f}")
        f5.metric("Profit Factor", f"{results_full['profit_factor']:.2f}")

        # Row 2: Execution & Validation
        st.markdown("#### Expectancy & Validation Summary")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("OOS Sharpe", f"{results_oos['sharpe']:.2f}", f"{oos_pct}% Split", delta_color="normal" if results_oos['sharpe'] > 0 else "inverse")
        e2.metric("Expectancy", f"{results_full['expectancy']:.4f}", help="Strategy value added per bar (Probability weighted outcomes).")
        e3.metric("Win Rate", f"{results_full['win_rate']:.1%}")
        e4.metric("Max Daily Gain", f"{results_full['max_win']:.2%}")
        e5.metric("Max Daily Loss", f"{results_full['max_loss']:.2%}")

        # Phase 21: Resonance Diagnostic HUD
        total_trades = np.sum(np.diff(signals) != 0)
        if total_trades == 0:
            st.warning(f"**Resonance Mismatch Detected**: Zero trades executed in the {periods_days[target_idx]}. Consider lowering the 'Resonance Threshold' or switching 'Signal Generation' mode to capture weaker precedence.")

        if results_oos['sharpe'] < 0.2:
            st.warning("Validation Divergence: Low Out-of-Sample metrics suggest potential overfitting at this frequency band.")

        # ── Equity and Drawdown Visualization ────────────────────────────────
        st.divider()
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            fig_eq = go.Figure()
            # Benchmark
            fig_eq.add_trace(go.Scatter(x=dates, y=results_full['market_curve'], name="Benchmark (B&H)", line=dict(color='rgba(255,255,255,0.2)', dash='dash')))
            # Full Equity
            fig_eq.add_trace(go.Scatter(x=dates, y=results_full['equity_curve'], name="Prime DSP (Full)", line=dict(color='#00ff41', width=2)))
            # OOS Highlight
            fig_eq.add_trace(go.Scatter(x=dates[oos_start:], y=results_full['equity_curve'][oos_start:], 
                                         name="Validation (OOS)", line=dict(color='#ffb300', width=3),
                                         fill='tozeroy', fillcolor='rgba(255, 179, 0, 0.05)'))
            
            fig_eq.add_vline(x=dates[oos_start], line_dash="dash", line_color="#ffb300")
            fig_eq.add_annotation(x=dates[oos_start], y=1.02, yref="paper", text="OOS START", showarrow=False, font=dict(color="#ffb300", size=10))
            fig_eq.update_layout(
                title="Cumulative Performance (In-Sample + OOS)", 
                yaxis_type="log", height=450, 
                template="plotly_dark", 
                yaxis_tickformat='.1%',
                font=dict(family="JetBrains Mono"),
                xaxis=dict(gridcolor='#1a1d21'),
                yaxis=dict(gridcolor='#1a1d21'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # Phase 21: Under-Water Drawdown HUD
        st.markdown("#### Risk Profile: Under-Water Drawdown")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dates, y=results_full['drawdown_curve'] * 100,
            fill='tozeroy', name="Drawdown",
            line=dict(color="#ff4b4b", width=1),
            hovertemplate='<b>Drawdown:</b> %{y:.2f}%<extra></extra>'
        ))
        fig_dd.update_layout(
            title="Equity Under-Water Profile (Peak-to-Trough %)",
            yaxis_title="Drawdown (%)",
            height=250, margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            font=dict(family="JetBrains Mono"),
            xaxis=dict(gridcolor='#1a1d21'),
            yaxis=dict(gridcolor='#1a1d21', range=[-max(1, abs(results_full['max_drawdown']*100)*1.2), 0.5])
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        with col_side:
            st.markdown("### Risk Attribution")
            st.metric("Aggregate Max DD", f"{results_full['max_drawdown']:.1%}")
            st.metric("Position Weight", f"{pos_size:.2f}x")

        # Expanded Drawdown Profile
        st.subheader("Drawdown Intensity Profile")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dates, y=results_full['drawdown_curve'], name="Full-Period DD", fill='tozeroy', line=dict(color='#ff4b4b')))
        fig_dd.add_trace(go.Scatter(x=dates[oos_start:], y=results_full['drawdown_curve'][oos_start:], name="OOS Phase DD", fill='tozeroy', line=dict(color='#ffb300', width=2)))
        fig_dd.update_layout(
            height=250, template="plotly_dark", yaxis_tickformat='.1%', 
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="JetBrains Mono"),
            xaxis=dict(gridcolor='#1a1d21'),
            yaxis=dict(gridcolor='#1a1d21')
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Signal Diagnostics ───────────────────────────────────────────────
        st.divider()
        st.subheader("Signal Logic Transparency")
        
        avg_phase = np.mean(pha_grid[target_idx])
        period    = 1.0 / freqs[target_idx] if freqs[target_idx] > 0 else 1
        lead_days = (avg_phase * period) / (2 * np.pi)

        c_l1, c_l2 = st.columns(2)
        with c_l1:
            st.write(f"**Primary Driver**: {signal_source}")
            st.write(f"**Detected Lead**: {abs(lead_days):.1f} Trading Days")
            st.info(f"Signal Strategy: {signal_mode}. Structural MA Filter: {'Active' if use_trend_filter else 'Disabled'}")
            
            if use_trend_filter:
                ma_series = pd.Series(prices).rolling(ma_period).mean()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=dates, y=prices, name=f"{traded_asset} Price", line=dict(color="#00ff41")))
                fig_trend.add_trace(go.Scatter(x=dates, y=ma_series.values, name=f"{ma_period}-day MA", line=dict(color="#ffb300", dash="dot")))
                fig_trend.update_layout(
                    title=f"Price vs {ma_period}-Day MA (Trend Filter)", 
                    height=280, template="plotly_dark",
                    font=dict(family="JetBrains Mono"),
                    xaxis=dict(gridcolor='#1a1d21'),
                    yaxis=dict(gridcolor='#1a1d21')
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with c_l2:
            fig_pha = go.Figure()
            fig_pha.add_trace(go.Scatter(x=dates, y=pha_grid[target_idx], name="Phase Angle", line=dict(color='#a78bfa')))
            fig_pha.update_layout(
                title="Spectral Information-Flow Topology",
                xaxis_title="Time Index",
                yaxis_title="Phase Angle (Radians)",
                height=450, margin=dict(l=0, r=0, t=30, b=0),
                template="plotly_dark",
                font=dict(family="JetBrains Mono"),
                xaxis=dict(gridcolor='#1a1d21'),
                yaxis=dict(gridcolor='#1a1d21'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pha, use_container_width=True)

        st.caption(
            "⚠️ The CWT is non-causal: metrics include lookahead leakage in the in-sample period. "
            "Validate all strategies using the Out-of-Sample (OOS) highlight above for live-tracking expectancy."
        )

    else:
        st.error("Select unique asset pair to begin simulation.")
