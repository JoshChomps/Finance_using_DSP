import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time as dtime
from engine.scalogram import track_frequency_flow
from engine.ui import inject_custom_css

st.set_page_config(page_title="Intraday Live Flow | Market DNA", layout="wide")
inject_custom_css(st)

st.title("Intraday Live Flow")
st.markdown("Real-time tracking of intraday volatility using Short-Time frequency analysis.")

stock     = st.sidebar.text_input("Intraday Symbol", value="SPY")
timeframe = st.sidebar.selectbox("Candle Aggregation", ["1m", "5m", "15m"], index=0)

st.sidebar.divider()
with st.sidebar.expander("Reference Intraday Symbols"):
    st.markdown("""
    **Indices and Broad Market**
    - `SPY` (S&P 500)
    - `QQQ` (Nasdaq 100)
    - `^VIX` (Volatility Index)
    
    **Equity Benchmarks**
    - `AAPL`, `MSFT`, `NVDA`
    - `TSLA`, `AMZN`, `META`
    
    **Digital Assets**
    - `BTC-USD`
    - `ETH-USD`
    """)


def _market_status():
    """Returns (is_open: bool, reason: str)."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        from datetime import timezone, timedelta
        now_et = datetime.now(timezone(timedelta(hours=-5)))

    weekday = now_et.weekday()
    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return False, f"Markets are closed on {day_name}."

    t = now_et.time()
    market_open  = dtime(9, 30)
    market_close = dtime(16, 0)

    if t < market_open:
        return False, f"Pre-market. US equity markets open at 09:30 ET (Current: {t.strftime('%H:%M')} ET)."
    if t > market_close:
        return False, f"Post-market. US equity markets closed at 16:00 ET (Current: {t.strftime('%H:%M')} ET)."

    return True, "Market active."


@st.cache_data(ttl=60)
def load_live_data(symbol, resolution):
    """Fetches intraday bars via provider."""
    prices = yf.download(symbol, period="5d", interval=resolution, progress=False)
    if prices.empty:
        return None
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    return prices


with st.spinner(f"Retrieving intraday data for {stock}:"):
    raw_data = load_live_data(stock, timeframe)

if raw_data is not None:
    close_col = raw_data["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    equity_path = close_col.values
    daily_diff  = np.log(equity_path[1:] / equity_path[:-1])
    daily_diff  = np.nan_to_num(daily_diff)

    st.subheader(f"Spectral Distribution: {stock}")

    block_size = 60 if timeframe == "1m" else 12
    freqs, times, intensity_z = track_frequency_flow(daily_diff, sample_rate=1.0, window_size=block_size)
    intensity = np.abs(intensity_z)

    mapped_times = raw_data.index[np.clip(np.round(times).astype(int), 0, len(raw_data)-1)]

    fig_stft = go.Figure(data=go.Heatmap(
        z=intensity,
        x=mapped_times,
        y=freqs,
        colorscale="Inferno",
        showscale=True,
        hovertemplate='<b>Time:</b> %{x|%I:%M %p}<br><b>Frequency:</b> %{y:.3f}<br><b>Volatility:</b> %{z:.3f}<extra></extra>'
    ))
    fig_stft.update_layout(
        title=f"Rolling frequency distribution ({timeframe} intervals)",
        xaxis_title="Timeline",
        yaxis_title="Frequency Scale",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_stft, width='stretch')

    # Price context
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        y=equity_path, mode="lines", name="Price",
        line=dict(color="#00ff9d"),
    ))
    fig_price.update_layout(
        title="Intraday Valuation Path", 
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_price, width='stretch')

    st.divider()
    with st.expander("Methodology Summary", expanded=True):
        st.markdown(f"""
        #### Intraday Frequency Analysis
        This module utilizes Short-Time frequency analysis to decompose intraday price action into its constituent spectral components. This provides a localized measurement of volatility distribution.

        **Data Interpretation:**
        - **Temporal Domain (X-Axis)**: Market time series timeline.
        - **Frequency Domain (Y-Axis)**: Rate of price variance. Higher values indicate high-frequency noise; lower values indicate underlying momentum.
        - **Spectral Intensity (Heat)**: Cumulative energy at a specific time-frequency coordinate.

        **Observed Patterns:**
        - **Impulse Bursts**: Vertical clusters signify broad-spectrum volatility, typically associated with market open or scheduled macroeconomic releases.
        - **High-Frequency Dominance**: Persistent energy in the upper frequency bands indicates algorithmic micro-volatility.
        - **Low-Frequency Dominance**: Energy concentration in lower bands suggests the formation of a structural intraday trend.
        """)

    st.info(
        "Technical Note: In production environments, this module interfaces with real-time "
        "WebSocket protocols for tick-level streaming."
    )

else:
    is_open, reason = _market_status()

    if timeframe == "1m" and not is_open:
        st.warning(
            f"1-minute resolution data availability is constrained to market hours.\n\n"
            f"{reason}\n\n"
            "Request 5m or 15m intervals for historical intraday analysis."
        )
    elif not is_open:
        st.warning(
            f"{reason}\n\n"
            "Historical intraday data remains available for the preceding five sessions."
        )
    else:
        st.error(
            f"Data retrieval failed for symbol: {stock}. "
            "Verify symbol accuracy and provider availability."
        )
