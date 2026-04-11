import streamlit as st
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
from engine.ui import inject_custom_css

st.set_page_config(page_title="Backtesting Simulator | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Backtesting Simulator")
st.markdown("Quantitative evaluation of trading strategies derived from cross-asset resonance and phase-shifted leadership signals.")

# Sidebar
st.sidebar.header("Asset Pair")
traded_asset  = st.sidebar.selectbox("Asset to Trade",           ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=0)
signal_source = st.sidebar.selectbox("Signal Source (Co-Asset)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=2)

st.sidebar.subheader("Signal Mode")
signal_mode = st.sidebar.radio(
    "Strategy Type",
    ["Phase-Following (Lead/Lag)", "Coherence Mean-Reversion"],
    help=(
        "Phase-Following: generates directional signals from the wavelet phase angle. "
        "Execution is conditional on the signal source exhibiting directional precedence. "
        "Coherence Mean-Reversion: generates reversion signals during instances of extreme frequency decoupling."
    ),
)

if signal_mode == "Phase-Following (Lead/Lag)":
    st.sidebar.subheader("Phase Parameters")
    coh_threshold   = st.sidebar.slider("Min Coherence to Trade", 0.4, 0.9, 0.6, 0.05)
    phase_strength  = st.sidebar.slider("Min Phase Strength (rad)", 0.1, 1.0, 0.3, 0.05,
                                         help="Minimum |phase| to confirm the source is leading.")
    phase_smoothing = st.sidebar.slider("Phase Smoothing (days)", 3, 20, 5)
else:
    st.sidebar.subheader("Resonance Thresholds")
    high_limit = st.sidebar.slider("Upper Limit (Sell/Short)", 0.5, 0.99, 0.7)
    low_limit  = st.sidebar.slider("Lower Limit (Buy/Long)",   0.01, 0.5,  0.3)

st.sidebar.subheader("Execution")
fee_perc = st.sidebar.number_input(
    "Transaction Fee (%)", min_value=0.0, max_value=2.0, value=0.05, step=0.01
) / 100.0

st.sidebar.subheader("Position Sizing")
sizing_mode = st.sidebar.radio(
    "Method",
    ["Auto – Kelly Criterion", "Manual"],
    help=(
        "**Auto** uses half-Kelly sizing derived from the strategy's own return history. "
        "Returns 0% when the strategy has no positive edge (and won't trade). "
        "**Manual** sets the capital fraction directly."
    ),
)
manual_size = st.sidebar.slider("Position Size (%)", 10, 100, 100) / 100.0 if sizing_mode == "Manual" else None

st.sidebar.subheader("Trend Filter")
use_trend_filter = st.sidebar.checkbox("Apply MA Trend Filter", value=True)
ma_period = st.sidebar.slider("MA Period (days)", 20, 200, 50) if use_trend_filter else None

st.sidebar.subheader("Validation")
oos_pct = st.sidebar.slider(
    "Out-of-Sample Test Split (%)", 10, 50, 30,
    help=(
        "Reservations of a specific data percentage for out-of-sample validation. "
        "In-sample data drives signal generation; out-of-sample data provides performance verification. "
        "Note: CWT is non-causal; results incorporate localized adjacent data."
    ),
)

# ── Main ───────────────────────────────────────────────────────────────────────
if traded_asset == signal_source:
    st.warning("Please choose two different assets for pairs analysis.")
else:
    price_data1 = get_data(traded_asset)
    price_data2 = get_data(signal_source)

    if price_data1 is not None and price_data2 is not None:
        min_size    = min(len(price_data1), len(price_data2))
        raw_rets    = calculate_returns(price_data1).tail(min_size)
        signal_rets = calculate_returns(price_data2).tail(min_size)

        sample_size   = 1000
        z_score1      = z_score_normalize(raw_rets).tail(sample_size).values
        z_score2      = z_score_normalize(signal_rets).tail(sample_size).values
        final_returns = raw_rets.tail(sample_size).values

        # Keep date index for x-axis, then extract values for computation
        close_price_series = price_data1["Close"].tail(min_size).tail(sample_size)
        dates        = close_price_series.index
        close_prices = close_price_series.values

        with st.spinner("Calculating coherence and phase..."):
            resonance_grid, phase_grid, coi, freqs, sig = calculate_coherence(z_score1, z_score2)
            
        period_labels = [f"{1/f:.0f}-day cycle ({f:.3f} Hz)" if f > 0 else "DC" for f in freqs]
        
        with st.sidebar.expander("Target Cycle Glossary"):
            st.markdown("""
            **Micro-Volatility (2-5 days):**
            Captures short-term market noise and high-frequency oscillations.
            
            **Swing Momentum (10-30 days):**
            Identifies intermediate mean-reversion and momentum shifts.
            
            **Macro-Structural (50-200 days):**
            Tracks long-term institutional capital flows and structural trends.
            """)
            
        selected_idx  = st.selectbox(
            "Target Cycle Band",
            range(len(freqs)),
            format_func=lambda i: period_labels[i],
            index=len(freqs) // 4,
            help="Select the specific frequency component to isolate for signal generation."
        )
        selected_period = 1.0 / freqs[selected_idx] if freqs[selected_idx] > 0 else 0

        # Coherence stability score for the selected band
        stab = coherence_stability(resonance_grid, selected_idx, window=50)
        recent_stability = float(np.nanmean(stab[-100:])) if len(stab) >= 100 else float(np.nanmean(stab))
        stability_label  = "Stable" if recent_stability < 0.15 else ("Moderate" if recent_stability < 0.25 else "Erratic")

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Analytical Period", f"~{selected_period:.0f} days", help="Estimated cycle duration in trading days.")
        col_info2.metric("Mean Coherence", f"{np.mean(resonance_grid[selected_idx]):.3f}", help="Average spectral coupling intensity.")
        col_info3.metric("Relationship Stability", stability_label,
                         help="Rolling variance of coherence. Lower values indicate more reliable spectral relationships.")

        # Signal generation
        st.divider()

        if signal_mode == "Phase-Following (Lead/Lag)":
            decisions = create_phase_signals(
                resonance_grid, phase_grid,
                source_returns=z_score2,
                band_idx=selected_idx,
                freqs=freqs,
                coherence_threshold=coh_threshold,
                min_phase_strength=phase_strength,
                smoothing=phase_smoothing,
            )
            avg_phase_at_band = float(np.mean(phase_grid[selected_idx]))
            lead_days         = avg_phase_at_band * selected_period / (2 * np.pi)
            if lead_days < -0.5:
                st.info(
                    f"Lead-Lag Identification: **{signal_source}** exhibits directional precedence over **{traded_asset}** by "
                    f"approximately **{abs(lead_days):.1f} days**. The engine is currently tracking lead-adjusted returns."
                )
            elif lead_days > 0.5:
                st.warning(
                    f"Inverse Leadership: {traded_asset} exhibits precedence over {signal_source} at this frequency "
                    f"(Lead: {lead_days:.1f}d). This configuration may yield unreliable signals."
                )
            else:
                st.info("Synchronous coupling detected: Assets are currently in-phase at this frequency.")
        else:
            decisions = create_signals_from_resonance(
                resonance_grid, selected_idx,
                high_barrier=high_limit, low_barrier=low_limit,
            )

        if use_trend_filter:
            pre_count  = int(np.sum(decisions != 0))
            decisions  = apply_trend_filter(decisions, close_prices, ma_period=ma_period)
            post_count = int(np.sum(decisions != 0))
            blocked    = pre_count - post_count
            if blocked > 0:
                st.info(
                    f"Trend Filter: Restricted **{blocked}** signals "
                    f"({blocked / max(pre_count, 1) * 100:.0f}%) that opposed the {ma_period}-day structural regime."
                )

        # ── Position sizing ────────────────────────────────────────────────────
        if sizing_mode == "Auto – Kelly Criterion":
            position_size = compute_kelly_fraction(final_returns, decisions)
            if position_size == 0.0:
                st.warning(
                    "Neutral Expectancy: The strategy exhibits no positive statistical edge under current parameters. Execution halted."
                )
        else:
            position_size = manual_size

        # ── OOS split ─────────────────────────────────────────────────────────
        oos_start = int(len(final_returns) * (1 - oos_pct / 100))
        ret_oos   = final_returns[oos_start:]
        sig_oos   = decisions[oos_start:]

        results_full = run_backtest(final_returns, decisions, slippage=fee_perc, position_size=position_size)
        results_oos  = run_backtest(ret_oos, sig_oos, slippage=fee_perc, position_size=position_size) \
                       if len(ret_oos) > 10 else None

        # Performance Metrics
        st.subheader("Institutional Risk Performance")
        m_row1 = st.columns(4)
        m_row1[0].metric("Cumulative Profit", f"{results_full['final_profit'] * 100:.2f}%")
        m_row1[1].metric("Annualized Return", f"{results_full['annual_return'] * 100:.2f}%")
        m_row1[2].metric("Profit Factor", f"{results_full['profit_factor']:.2f}", help="Gross Profit / Gross Loss.")
        m_row1[3].metric("Win Rate", f"{results_full['win_rate'] * 100:.1f}%")
        
        m_row2 = st.columns(4)
        m_row2[0].metric("Sharpe Ratio", f"{results_full['sharpe']:.2f}", help="Risk-adjusted return penalizing total volatility.")
        m_row2[1].metric("Sortino Ratio", f"{results_full['sortino']:.2f}", help="Risk-adjusted return penalizing only downside volatility.")
        m_row2[2].metric("Calmar Ratio", f"{results_full['calmar']:.2f}", help="Annualized Return / Maximum Drawdown.")
        m_row2[3].metric("Max Drawdown", f"{results_full['max_drawdown'] * 100:.2f}%")

        if results_oos:
            st.subheader(f"Out-of-Sample Validation ({oos_pct}% Split)")
            o_row = st.columns(4)
            o_row[0].metric("OOS Profit", f"{results_oos['final_profit'] * 100:.2f}%")
            o_row[1].metric("OOS Sharpe", f"{results_oos['sharpe']:.2f}")
            o_row[2].metric("OOS Sortino", f"{results_oos['sortino']:.2f}")
            o_row[3].metric("OOS Max DD", f"{results_oos['max_drawdown'] * 100:.2f}%")

            if results_oos['sharpe'] < 0.3:
                st.warning("Low OOS Efficiency: Performance variance detected in the validation set. Potential overfitting.")

        # ── Equity curves ──────────────────────────────────────────────────────
        st.divider()
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=dates, y=results_full["market_curve"], name="Benchmark (B&H)",
            line=dict(color="rgba(128,128,128,0.5)", dash="dash"),
        ))
        fig_curve.add_trace(go.Scatter(
            x=dates, y=results_full["equity_curve"], name="DSP Strategy",
            line=dict(color="#00ff9d", width=2.5),
        ))
        
        # Add shading for active positions
        active_indices = np.where(decisions != 0)[0]
        if len(active_indices) > 0:
            # Group consecutive indices to create shaded regions
            diffs = np.diff(active_indices)
            breaks = np.where(diffs > 1)[0]
            start_indices = np.concatenate([[active_indices[0]], active_indices[breaks + 1]])
            end_indices = np.concatenate([active_indices[breaks], [active_indices[-1]]])
            
            for start, end in zip(start_indices, end_indices):
                fig_curve.add_vrect(
                    x0=dates[start], x1=dates[end],
                    fillcolor="rgba(0,255,157,0.05)", line_width=0, layer="below"
                )

        fig_curve.update_layout(
            title="Portfolio Valuation Path",
            yaxis_title="Cumulative Return (%)",
            height=500, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_curve, width='stretch')

        # Drawdown Underlay
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dates, y=results_full["drawdown_curve"] * 100,
            fill='tozeroy', name="Drawdown", line=dict(color="rgba(255, 75, 75, 0.7)")
        ))
        fig_dd.update_layout(
            title="Maximum Drawdown Profile (%)",
            yaxis_title="Variance (%)",
            height=350, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dd, width='stretch')

        if signal_mode == "Phase-Following (Lead/Lag)":
            with st.expander("Lead Logic Visualization"):
                st.markdown(f"""
                **Synchronization Profile:**
                - **Cycle Period**: ~{selected_period:.1f} trading days.
                - **Calculated Lead ($L$)**: {abs(lead_days):.1f} days (Signal Source leads).
                - **Anchor Point**: The engine is looking at **Source Asset (t - {int(abs(lead_days))})** 
                to determine the momentum bias for **Traded Asset (t + 1)**.
                
                This creates a 'preview' window where the leading asset's spectral phase 
                predicts the following asset's directional stance.
                """)

        st.divider()
        st.subheader("Signal Diagnostics")

        # Coherence
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=dates, y=resonance_grid[selected_idx, :], name="Resonance", line=dict(color="orange")))
        if signal_mode == "Coherence Mean-Reversion":
            fig_sig.add_hline(y=high_limit, line_dash="dash", line_color="red")
            fig_sig.add_hline(y=low_limit,  line_dash="dash", line_color="green")
        fig_sig.update_layout(
            title=f"Spectral Coupling Intensity (at {selected_period:.1f}-day periodicity)",
            height=300, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sig, width='stretch')

        # Phase
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(x=dates, y=phase_grid[selected_idx, :], name="Phase Angle", line=dict(color="#a78bfa")))
        fig_phase.add_hline(y=0, line_color="white", line_width=1)
        fig_phase.update_layout(
            title=f"Instantaneous Phase Angle (Negative = {signal_source} Precedence)",
            yaxis_title="Radians", height=250, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_phase, width='stretch')

        st.divider()
        with st.expander("Analytical Framework", expanded=True):
            st.markdown(f"""
            #### Lead-Lag DSP Methodology
            This simulator implements a cross-spectral leadership model to isolate alpha at specific market rhythms.
            
            **Analytical Components:**
            1. **Frequency De-Noising**: Markets are a superposition of cycles. We isolate the {selected_period:.1f}-day component to filter noise.
            2. **Spectral Coupling (Coherence)**: Measures the statistical reliability of the relationship between {traded_asset} and {signal_source}.
            3. **Phase-Shift Analysis**: Identifies directional precedence. When the phase is negative, {signal_source} acts as a predictive beacon for {traded_asset}.
            4. **Lead-Adjusted Execution**: The engine calculates the precise hourly lead and aligns trades with historical momentum shifts in the leader.
            """)

        st.caption("Disclaimer: DSP models incorporate localized adjacent data. Out-of-sample validation provides a necessary floor for production expectancy.")

    else:
        st.error("Data retrieval error: Unable to initialize price history.")
