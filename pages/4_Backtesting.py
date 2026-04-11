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

st.set_page_config(page_title="Backtesting Simulator | FinSignal Suite", layout="wide")
inject_custom_css(st)

st.title("📈 Backtesting Simulator")
st.markdown("Test trading strategies derived from cross-asset resonance and phase signals.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Asset Pair")
traded_asset  = st.sidebar.selectbox("Asset to Trade",          ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=0)
signal_source = st.sidebar.selectbox("Signal Source (Co-Asset)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=2)

st.sidebar.subheader("Signal Mode")
signal_mode = st.sidebar.radio(
    "Strategy Type",
    ["Phase-Following (Lead/Lag)", "Coherence Mean-Reversion"],
    help=(
        "**Phase-Following**: generates directional signals from the wavelet phase angle. "
        "Only trades when the signal source demonstrably leads the traded asset. "
        "This is the financially sound approach.\n\n"
        "**Coherence Mean-Reversion**: bets on reversion when coherence is extreme. "
        "Simpler but has no directional basis — treats all decoupling as identical."
    ),
)

if signal_mode == "Phase-Following (Lead/Lag)":
    st.sidebar.subheader("Phase Parameters")
    coh_threshold    = st.sidebar.slider("Min Coherence to Trade", 0.4, 0.9, 0.6, 0.05)
    phase_strength   = st.sidebar.slider("Min Phase Strength (rad)", 0.1, 1.0, 0.3, 0.05,
                                          help="Minimum |phase| to confirm the source is leading.")
    phase_smoothing  = st.sidebar.slider("Phase Smoothing (days)", 3, 20, 5)
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
        "Reserve this percentage of the data as a held-out test set. "
        "In-sample is used for signal generation; out-of-sample shows honest performance. "
        "Note: CWT is non-causal, so even in-sample numbers are optimistic."
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
        close_prices  = price_data1["Close"].tail(min_size).tail(sample_size).values

        with st.spinner("Calculating coherence and phase..."):
            resonance_grid, phase_grid, coi, freqs, sig = calculate_coherence(z_score1, z_score2)

        # Band selector — labelled in trading days so the user knows what they're picking
        period_labels = [f"~{1/f:.0f}-day cycle ({f:.3f} Hz)" if f > 0 else "DC" for f in freqs]
        selected_idx  = st.selectbox(
            "Select Cycle Band to Trade",
            range(len(freqs)),
            format_func=lambda i: period_labels[i],
            index=len(freqs) // 4,
        )
        selected_period = 1.0 / freqs[selected_idx] if freqs[selected_idx] > 0 else 0

        # Coherence stability score for the selected band
        stab = coherence_stability(resonance_grid, selected_idx, window=50)
        recent_stability = float(np.nanmean(stab[-100:])) if len(stab) >= 100 else float(np.nanmean(stab))
        stability_label  = "Stable ✓" if recent_stability < 0.15 else ("Moderate ⚠" if recent_stability < 0.25 else "Erratic ✗")

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Selected Cycle Period", f"~{selected_period:.0f} days")
        col_info2.metric("Avg Coherence at Band", f"{np.mean(resonance_grid[selected_idx]):.3f}")
        col_info3.metric("Coherence Stability", stability_label,
                         help="Rolling std of coherence. Low = reliable relationship at this frequency.")

        # ── Build signals ──────────────────────────────────────────────────────
        st.divider()

        if signal_mode == "Phase-Following (Lead/Lag)":
            decisions = create_phase_signals(
                resonance_grid, phase_grid,
                source_returns=z_score2,
                band_idx=selected_idx,
                coherence_threshold=coh_threshold,
                min_phase_strength=phase_strength,
                smoothing=phase_smoothing,
            )
            avg_phase_at_band = float(np.mean(phase_grid[selected_idx]))
            lead_days         = avg_phase_at_band * selected_period / (2 * np.pi)
            if lead_days < -0.5:
                st.info(
                    f"Phase analysis: **{signal_source}** leads **{traded_asset}** by "
                    f"approximately **{abs(lead_days):.1f} days** at the ~{selected_period:.0f}-day cycle. "
                    f"Signals follow {signal_source}'s trend direction."
                )
            elif lead_days > 0.5:
                st.warning(
                    f"Phase analysis: **{traded_asset}** leads **{signal_source}** at this cycle "
                    f"(lead ≈ {lead_days:.1f}d). The source asset is the follower here — "
                    f"consider swapping the pair or choosing a different band."
                )
            else:
                st.info("Assets are approximately in phase at this cycle (no clear leader).")
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
                    f"Trend filter blocked **{blocked}** signal(s) "
                    f"({blocked / max(pre_count, 1) * 100:.0f}%) that opposed the "
                    f"{ma_period}-day moving average."
                )

        # ── Position sizing ────────────────────────────────────────────────────
        if sizing_mode == "Auto – Kelly Criterion":
            position_size = compute_kelly_fraction(final_returns, decisions)
            if position_size == 0.0:
                st.warning(
                    "Kelly returned 0% — the strategy shows no positive expectancy "
                    "on this data. Try a different cycle band, signal mode, or asset pair."
                )
        else:
            position_size = manual_size

        # ── OOS split ─────────────────────────────────────────────────────────
        oos_start = int(len(final_returns) * (1 - oos_pct / 100))
        ret_is    = final_returns[:oos_start]
        ret_oos   = final_returns[oos_start:]
        sig_is    = decisions[:oos_start]
        sig_oos   = decisions[oos_start:]

        results_full = run_backtest(final_returns, decisions, slippage=fee_perc, position_size=position_size)
        results_oos  = run_backtest(ret_oos, sig_oos, slippage=fee_perc, position_size=position_size) \
                       if len(ret_oos) > 10 else None

        # ── Metric cards ───────────────────────────────────────────────────────
        st.subheader("Full-Period Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Profit",   f"{results_full['final_profit'] * 100:.2f}%",
                  f"B&H: {results_full['market_curve'][-1] * 100:.2f}%")
        c2.metric("Annual Return",  f"{results_full['annual_return'] * 100:.2f}%")
        c3.metric("Sharpe",         f"{results_full['sharpe']:.2f}")
        c4.metric("Max Drawdown",   f"{results_full['max_drawdown'] * 100:.2f}%")
        c5.metric("Position Size",  f"{position_size * 100:.0f}%",
                  "Kelly (half)" if sizing_mode == "Auto – Kelly Criterion" else "Manual")

        if results_oos:
            st.subheader(f"Out-of-Sample ({oos_pct}% held out) — honest estimate")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("OOS Profit",       f"{results_oos['final_profit'] * 100:.2f}%",
                      f"B&H: {results_oos['market_curve'][-1] * 100:.2f}%")
            o2.metric("OOS Annual Return", f"{results_oos['annual_return'] * 100:.2f}%")
            o3.metric("OOS Sharpe",        f"{results_oos['sharpe']:.2f}")
            o4.metric("OOS Max Drawdown",  f"{results_oos['max_drawdown'] * 100:.2f}%")

            if results_oos['sharpe'] < 0.3:
                st.warning(
                    "OOS Sharpe is low — the strategy may be overfitting the in-sample "
                    "period. Consider a more conservative position size or different band."
                )

        # ── Equity curves ──────────────────────────────────────────────────────
        st.divider()
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            y=results_full["market_curve"], name="Buy & Hold",
            line=dict(color="gray", dash="dash"),
        ))
        fig_curve.add_trace(go.Scatter(
            y=results_full["equity_curve"], name="DSP Strategy (full)",
            line=dict(color="#00ff9d", width=2),
        ))
        if results_oos:
            # Show OOS portion as a highlighted region
            oos_x = list(range(oos_start, len(final_returns)))
            fig_curve.add_trace(go.Scatter(
                x=oos_x, y=results_full["equity_curve"][oos_start:],
                name=f"OOS region ({oos_pct}%)",
                line=dict(color="#ffd700", width=2),
                fill="tozeroy", fillcolor="rgba(255,215,0,0.07)",
            ))
        fig_curve.add_vline(x=oos_start, line_dash="dash", line_color="gold",
                            annotation_text="OOS Start", annotation_position="top left")
        fig_curve.update_layout(
            title="Portfolio Growth — In-Sample vs Out-of-Sample",
            yaxis_title="Profit / Loss",
            xaxis_title="Trading Days",
            height=500,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # ── Signal visualization ───────────────────────────────────────────────
        st.divider()
        st.subheader("Signal Visualization")

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            y=resonance_grid[selected_idx, :], name="Coherence",
            line=dict(color="orange"),
        ))
        if signal_mode == "Coherence Mean-Reversion":
            fig_sig.add_hline(y=high_limit, line_dash="dash", line_color="red",   annotation_text="Sell Barrier")
            fig_sig.add_hline(y=low_limit,  line_dash="dash", line_color="green", annotation_text="Buy Barrier")
        fig_sig.update_layout(
            title=f"Coherence at ~{selected_period:.0f}-day cycle",
            height=280,
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        # Phase angle chart
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            y=phase_grid[selected_idx, :], name="Phase Angle",
            line=dict(color="#a78bfa"),
        ))
        fig_phase.add_hline(y=0, line_color="white", line_width=1)
        fig_phase.update_layout(
            title=f"Phase Angle at ~{selected_period:.0f}-day cycle "
                  f"(negative = {signal_source} leads {traded_asset})",
            yaxis_title="Radians",
            height=250,
        )
        st.plotly_chart(fig_phase, use_container_width=True)

        # Trend overlay
        if use_trend_filter:
            ma_series = pd.Series(close_prices).rolling(ma_period).mean()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                y=close_prices, name=f"{traded_asset} Price",
                line=dict(color="#4fa3e0"),
            ))
            fig_trend.add_trace(go.Scatter(
                y=ma_series.values, name=f"{ma_period}-day MA",
                line=dict(color="gold", dash="dot"),
            ))
            fig_trend.update_layout(title=f"Price vs {ma_period}-Day MA (Trend Filter)", height=280)
            st.plotly_chart(fig_trend, use_container_width=True)

        st.caption(
            "⚠️ The CWT is non-causal: coherence and phase at time *t* are influenced by "
            "neighbouring future bars. The backtest numbers are structurally optimistic. "
            "The OOS split above gives a more realistic estimate, but a proper "
            "walk-forward test using a causal (one-sided) wavelet would be needed for live deployment."
        )

    else:
        st.error("Ran into trouble loading the price history.")
