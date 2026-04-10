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
    compute_kelly_fraction,
    apply_trend_filter,
)
from engine.ui import inject_custom_css

st.set_page_config(page_title="Backtesting Simulator | FinSignal Suite", layout="wide")
inject_custom_css(st)

st.title("📈 Backtesting Simulator")
st.markdown("Test trading strategies derived from cross-asset resonance signals.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Strategy Settings")
traded_asset  = st.sidebar.selectbox("Asset to Trade",        ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=0)
signal_source = st.sidebar.selectbox("Co-Asset (Signal Source)", ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC-USD"], index=2)

st.sidebar.subheader("Logic Thresholds")
high_limit = st.sidebar.slider("Upper Limit (Neutral/Sell)", 0.5, 0.99, 0.7)
low_limit  = st.sidebar.slider("Lower Limit (Buy/Long)",    0.01, 0.5,  0.3)
fee_perc   = st.sidebar.number_input(
    "Transaction Fee (%)", min_value=0.0, max_value=2.0, value=0.05, step=0.01
) / 100.0

st.sidebar.subheader("Position Sizing")
sizing_mode = st.sidebar.radio(
    "Method",
    ["Auto – Kelly Criterion", "Manual"],
    help=(
        "**Auto** computes the theoretically optimal fraction of capital to "
        "deploy per trade using the Kelly Criterion (half-Kelly for safety). "
        "**Manual** lets you set the fraction directly."
    ),
)
if sizing_mode == "Manual":
    manual_size = st.sidebar.slider("Position Size (% of capital)", 10, 100, 100) / 100.0
else:
    manual_size = None  # will be computed below

st.sidebar.subheader("Trend Filter")
use_trend_filter = st.sidebar.checkbox(
    "Apply MA Trend Filter",
    value=True,
    help=(
        "Prevents the strategy from fighting the macro trend. "
        "Long signals are blocked when price is below its moving average; "
        "short signals are blocked when price is above it."
    ),
)
if use_trend_filter:
    ma_period = st.sidebar.slider("MA Period (days)", 20, 200, 50)

# ── Main ───────────────────────────────────────────────────────────────────────
if traded_asset == signal_source:
    st.warning("Please choose two different assets for pairs analysis.")
else:
    price_data1 = get_data(traded_asset)
    price_data2 = get_data(signal_source)

    if price_data1 is not None and price_data2 is not None:
        # Align timelines
        min_size = min(len(price_data1), len(price_data2))
        raw_rets    = calculate_returns(price_data1).tail(min_size)
        signal_rets = calculate_returns(price_data2).tail(min_size)

        sample_size = 1000
        z_score1      = z_score_normalize(raw_rets).tail(sample_size).values
        z_score2      = z_score_normalize(signal_rets).tail(sample_size).values
        final_returns = raw_rets.tail(sample_size).values

        # Price series for the trend filter (last sample_size rows)
        close_prices = price_data1["Close"].tail(min_size).tail(sample_size).values

        with st.spinner("Analyzing resonance..."):
            resonance_grid, phase, coi, freqs, sig = calculate_coherence(z_score1, z_score2)

        st.subheader(f"Strategy: Reverting {traded_asset} based on {signal_source} Resonance")

        # Pick the frequency band to trade
        choices      = [f"Band {i}: {f:.3f} Hz" for i, f in enumerate(freqs)]
        selected_idx = st.selectbox(
            "Select Cycle Band",
            range(len(freqs)),
            format_func=lambda i: choices[i],
            index=len(freqs) // 4,
        )

        # ── Build signals ──────────────────────────────────────────────────────
        decisions = create_signals_from_resonance(
            resonance_grid, selected_idx,
            high_barrier=high_limit, low_barrier=low_limit,
        )

        if use_trend_filter:
            pre_filter_count = int(np.sum(decisions != 0))
            decisions = apply_trend_filter(decisions, close_prices, ma_period=ma_period)
            post_filter_count = int(np.sum(decisions != 0))
            blocked = pre_filter_count - post_filter_count
            if blocked > 0:
                st.info(
                    f"Trend filter blocked **{blocked}** signal(s) "
                    f"({blocked / max(pre_filter_count, 1) * 100:.0f}%) that went against the "
                    f"{ma_period}-day moving average."
                )

        # ── Position sizing ────────────────────────────────────────────────────
        if sizing_mode == "Auto – Kelly Criterion":
            position_size = compute_kelly_fraction(final_returns, decisions)
        else:
            position_size = manual_size

        # ── Run backtest ───────────────────────────────────────────────────────
        results = run_backtest(final_returns, decisions, slippage=fee_perc, position_size=position_size)

        # ── Metric cards ───────────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Profit",    f"{results['final_profit'] * 100:.2f}%",
                    f"B&H: {results['market_curve'][-1] * 100:.2f}%")
        col2.metric("Annual Returns",  f"{results['annual_return'] * 100:.2f}%")
        col3.metric("Sharpe",          f"{results['sharpe']:.2f}")
        col4.metric("Worst Drawdown",  f"{results['max_drawdown'] * 100:.2f}%")
        col5.metric(
            "Position Size",
            f"{position_size * 100:.0f}%",
            "Kelly" if sizing_mode == "Auto – Kelly Criterion" else "Manual",
        )

        # ── Growth chart ───────────────────────────────────────────────────────
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            y=results["market_curve"],
            name="Buy & Hold",
            line=dict(color="gray", dash="dash"),
        ))
        fig_curve.add_trace(go.Scatter(
            y=results["equity_curve"],
            name="DSP Strategy",
            line=dict(color="#00ff9d", width=2),
        ))
        fig_curve.update_layout(
            title="Portfolio Growth Over Time",
            yaxis_title="Profit / Loss",
            xaxis_title="Time index",
            height=500,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # ── Signal visualization ───────────────────────────────────────────────
        st.divider()
        st.subheader("Signal Visualization")

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            y=resonance_grid[selected_idx, :],
            name="Asset Resonance",
            line=dict(color="orange"),
        ))
        fig_sig.add_hline(y=high_limit, line_dash="dash", line_color="red",   annotation_text="Upper Barrier")
        fig_sig.add_hline(y=low_limit,  line_dash="dash", line_color="green", annotation_text="Lower Barrier")
        fig_sig.update_layout(
            title=f"Coherence Level at {freqs[selected_idx]:.3f} Hz",
            height=300,
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        # Price vs MA overlay when trend filter is active
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
            fig_trend.update_layout(
                title=f"Price vs {ma_period}-Day Moving Average (Trend Filter)",
                height=300,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    else:
        st.error("Ran into trouble loading the price history.")
