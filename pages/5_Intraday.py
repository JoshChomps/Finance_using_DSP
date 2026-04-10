import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time as dtime
from engine.scalogram import track_frequency_flow
from engine.ui import inject_custom_css

st.set_page_config(page_title="Intraday Live Flow | FinSignal Suite", layout="wide")
inject_custom_css(st)

st.title("📡 Intraday Live Flow")
st.markdown("Real-time tracking of intraday volatility using Short-Time frequency analysis.")

stock     = st.sidebar.text_input("Intraday Symbol (e.g. SPY, ^VIX)", value="SPY")
timeframe = st.sidebar.selectbox("Candle Aggregation", ["1m", "5m", "15m"], index=0)


def _market_status():
    """
    Returns (is_open: bool, reason: str).
    Uses UTC directly to avoid needing tzdata/pytz — ET is UTC-5 (winter) / UTC-4 (summer).
    We approximate by checking both windows and being conservative.
    """
    try:
        # Python 3.9+ stdlib; no extra package needed on most systems
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # Fallback: approximate ET as UTC-5 (conservative — may miss last hour in summer)
        from datetime import timezone, timedelta
        now_et = datetime.now(timezone(timedelta(hours=-5)))

    weekday = now_et.weekday()   # 0=Mon … 6=Sun
    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return False, f"Markets are closed on {day_name}. Come back Monday."

    t = now_et.time()
    market_open  = dtime(9, 30)
    market_close = dtime(16, 0)

    if t < market_open:
        return False, f"Pre-market. US equity markets open at 09:30 ET (currently {t.strftime('%H:%M')} ET)."
    if t > market_close:
        return False, f"After-hours. US equity markets closed at 16:00 ET (currently {t.strftime('%H:%M')} ET)."

    return True, "Market is open."


@st.cache_data(ttl=60)  # Cache for 60 seconds — avoid hammering the API during dev
def load_live_data(symbol, resolution):
    """Fetches a week of intraday bars and returns a flat-column DataFrame or None."""
    prices = yf.download(symbol, period="5d", interval=resolution, progress=False)
    if prices.empty:
        return None
    # Flatten MultiIndex columns from newer yfinance versions
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    return prices


with st.spinner(f"Fetching live data for {stock}..."):
    raw_data = load_live_data(stock, timeframe)

if raw_data is not None:
    # Close is always a plain 1-D series after the flatten above, but guard anyway
    close_col = raw_data["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    equity_path = close_col.values
    daily_diff  = np.log(equity_path[1:] / equity_path[:-1])
    daily_diff  = np.nan_to_num(daily_diff)

    st.subheader(f"Current Structure: {stock}")

    # Window size adapts to the candle resolution
    block_size = 60 if timeframe == "1m" else 12
    freqs, times, intensity_z = track_frequency_flow(daily_diff, sample_rate=1.0, window_size=block_size)
    intensity = np.abs(intensity_z)

    # Intraday spectrogram
    fig_stft = go.Figure(data=go.Heatmap(
        z=intensity,
        x=times,
        y=freqs,
        colorscale="Inferno",
        showscale=True,
    ))
    fig_stft.update_layout(
        title=f"Rolling frequency map ({timeframe} candles)",
        xaxis_title="Time Blocks",
        yaxis_title="Relative Frequency",
        height=500,
    )
    st.plotly_chart(fig_stft, use_container_width=True)

    # Price line for context
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        y=equity_path, mode="lines", name="Price",
        line=dict(color="#00ff9d"),
    ))
    fig_price.update_layout(title="Intraday Price History", height=300)
    st.plotly_chart(fig_price, use_container_width=True)

    st.info(
        "💡 **Pro Tip**: In a production setup this would be wired to a real-time "
        "WebSocket for tick-by-tick streaming."
    )

else:
    # Give a specific, actionable error instead of the generic one
    is_open, reason = _market_status()

    if timeframe == "1m" and not is_open:
        st.warning(
            f"**1-minute data is only available during market hours.**\n\n"
            f"{reason}\n\n"
            "Try switching to **5m** or **15m** — Yahoo Finance keeps several "
            "days of those even after close."
        )
    elif not is_open:
        st.warning(
            f"{reason}\n\n"
            "Recent intraday data (5m / 15m) is usually still available for the "
            "last few trading days even outside market hours."
        )
    else:
        st.error(
            f"Couldn't load intraday data for **{stock}**. "
            "Double-check the ticker symbol and try again."
        )
