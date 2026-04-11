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
st.markdown("Test trading strategies derived from cross-asset resonance and phase signals.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Asset Pair")
traded_asset  = st.sidebar.selectbox("Asset to Trade",           ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=0)
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

        # Keep date index for x-axis, then extract values for computation
        close_price_series = price_data1["Close"].tail(min_size).tail(sample_size)
        dates        = close_price_series.index
        close_prices = close_price_series.values

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
        ret_oos   = final_returns[oos_start:]
        sig_oos   = decisions[oos_start:]

        results_full = run_backtest(final_returns, decisions, slippage=fee_perc, position_size=position_size)
        results_oos  = run_backtest(ret_oos, sig_oos, slippage=fee_perc, position_size=position_size) \
                       if len(ret_oos) > 10 else None

        # ── Metric cards ───────────────────────────────────────────────────────
        st.subheader("Full-Period Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Profit",  f"{results_full['final_profit'] * 100:.2f}%",
                  f"B&H: {results_full['market_curve'][-1] * 100:.2f}%")
        c2.metric("Annual Return", f"{results_full['annual_return'] * 100:.2f}%")
        c3.metric("Sharpe",        f"{results_full['sharpe']:.2f}")
        c4.metric("Max Drawdown",  f"{results_full['max_drawdown'] * 100:.2f}%")
        c5.metric("Position Size", f"{position_size * 100:.0f}%",
                  "Kelly (half)" if sizing_mode == "Auto – Kelly Criterion" else "Manual")

        if results_oos:
            st.subheader(f"Out-of-Sample ({oos_pct}% held out) — honest estimate")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("OOS Profit",        f"{results_oos['final_profit'] * 100:.2f}%",
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
            x=dates,
            y=results_full["market_curve"],
            name="Buy & Hold",
            line=dict(color="gray", dash="dash"),
            hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>B&H P/L:</b> %{y:.2%}<extra></extra>'
        ))
        fig_curve.add_trace(go.Scatter(
            x=dates,
            y=results_full["equity_curve"],
            name="DSP Strategy",
            line=dict(color="#00ff9d", width=2),
            hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Strategy P/L:</b> %{y:.2%}<extra></extra>'
        ))
        if results_oos and oos_start < len(dates):
            oos_dates = dates[oos_start:]
            fig_curve.add_trace(go.Scatter(
                x=oos_dates,
                y=results_full["equity_curve"][oos_start:],
                name=f"OOS region ({oos_pct}%)",
                line=dict(color="#ffd700", width=2),
                fill="tozeroy", fillcolor="rgba(255,215,0,0.07)",
            ))
        if oos_start < len(dates):
            fig_curve.add_vline(
                x=str(dates[oos_start]),
                line_dash="dash", line_color="gold",
                annotation_text="OOS Start", annotation_position="top left"
            )
        fig_curve.update_layout(
            title="Portfolio Growth — In-Sample vs Out-of-Sample",
            yaxis_title="Profit / Loss",
            xaxis_title="Date",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_curve, width='stretch')

        # ── Signal visualization ───────────────────────────────────────────────
        st.divider()
        st.subheader("Signal Visualization")

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=dates,
            y=resonance_grid[selected_idx, :],
            name="Asset Resonance",
            line=dict(color="orange"),
            hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Coherence Level:</b> %{y:.4f}<extra></extra>'
        ))
        if signal_mode == "Coherence Mean-Reversion":
            fig_sig.add_hline(y=high_limit, line_dash="dash", line_color="red",   annotation_text="Sell Barrier")
            fig_sig.add_hline(y=low_limit,  line_dash="dash", line_color="green", annotation_text="Buy Barrier")
        fig_sig.update_layout(
            title=f"Coherence at ~{selected_period:.0f}-day cycle ({freqs[selected_idx]:.3f} Hz)",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_sig, width='stretch')

        # Phase angle chart
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=dates,
            y=phase_grid[selected_idx, :],
            name="Phase Angle",
            line=dict(color="#a78bfa"),
            hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Phase:</b> %{y:.3f} rad<extra></extra>'
        ))
        fig_phase.add_hline(y=0, line_color="white", line_width=1)
        fig_phase.update_layout(
            title=f"Phase Angle at ~{selected_period:.0f}-day cycle "
                  f"(negative = {signal_source} leads {traded_asset})",
            yaxis_title="Radians",
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_phase, width='stretch')

        # Trend overlay
        if use_trend_filter:
            ma_series = pd.Series(close_prices).rolling(ma_period).mean()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=dates, y=close_prices, name=f"{traded_asset} Price",
                line=dict(color="#4fa3e0"),
                hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
            ))
            fig_trend.add_trace(go.Scatter(
                x=dates, y=ma_series.values, name=f"{ma_period}-day MA",
                line=dict(color="gold", dash="dot"),
                hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>MA:</b> $%{y:.2f}<extra></extra>'
            ))
            fig_trend.update_layout(
                title=f"Price vs {ma_period}-Day Moving Average (Trend Filter)",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_trend, width='stretch')

        st.divider()
        with st.expander("📖 Interpreting the Backtest Results", expanded=True):
            st.markdown(f"""
            #### Qualitative Summary
            This simulator tests the viability of trading **{traded_asset}** by measuring its
            phase-alignment (resonance) with **{signal_source}**.

            **What is it doing?**
            1. It isolates a specific frequency band (e.g., Weekly cycles).
            2. It tracks the resonance between the two assets at that exact frequency.
            3. In **Phase-Following** mode: signals are generated only when {signal_source}
               demonstrably leads {traded_asset}, following its trend direction.
            4. In **Mean-Reversion** mode: buys when coherence drops below the lower barrier
               (anticipating re-coupling) and closes when coherence hits the upper barrier.

            **Understanding the Metrics:**
            - **Total Profit vs B&H**: Shows if active trading outperformed just holding the asset.
            - **Sharpe Ratio**: Measures risk-adjusted return. A Sharpe > 1.0 is good, > 1.5 is excellent.
            - **Worst Drawdown**: The maximum drop from peak to trough.
            - **Position Size**: If set to Auto, the engine calculates the half-Kelly criterion.
              For safety, the engine caps at a maximum of `100%` (no leverage).
            - **OOS Sharpe**: The honest estimate — computed on data the model never saw.
            """)

        st.caption(
            "⚠️ The CWT is non-causal: coherence and phase at time *t* are influenced by "
            "neighbouring future bars. The backtest numbers are structurally optimistic. "
            "The OOS split above gives a more realistic estimate, but a proper "
            "walk-forward test using a causal (one-sided) wavelet would be needed for live deployment."
        )

    else:
        st.error("Ran into trouble loading the price history.")
