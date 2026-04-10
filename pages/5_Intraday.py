import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from engine.scalogram import track_frequency_flow
from engine.ui import inject_custom_css

st.set_page_config(page_title="Intraday Live Flow | FinSignal Suite", layout="wide")
inject_custom_css(st)

st.title("📡 Intraday Live Flow")
st.markdown("Real-time tracking of intraday volatility using Short-Time frequency analysis.")

stock = st.sidebar.text_input("Intraday Symbol (e.g. SPY, ^VIX)", value="SPY")
timeframe = st.sidebar.selectbox("Candel Aggregation", ["1m", "5m", "15m"], index=0)

@st.cache_data(ttl=60) # Cache for 60 seconds so we don't spam the API during development
def load_live_data(symbol, resolution):
    # Get a week's worth of intraday bars
    prices = yf.download(symbol, period="5d", interval=resolution)
    if prices.empty:
        return None
    return prices

with st.spinner(f"Fetching live numbers for {stock}..."):
    raw_data = load_live_data(stock, timeframe)

if raw_data is not None:
    # Get the basic price series
    equity_path = raw_data['Close'].values
    daily_diff = np.log(equity_path[1:] / equity_path[:-1])
    daily_diff = np.nan_to_num(daily_diff)
    
    st.subheader(f"Current Structure: {stock}")
    
    # Calculate frequency flow (STFT)
    # nperseg should adapt based on the candle timeframe
    block_size = 60 if timeframe == '1m' else 12 
    freqs, times, intensity_z = track_frequency_flow(daily_diff, sample_rate=1.0, window_size=block_size)
    intensity = np.abs(intensity_z)
    
    # Intraday Spectrogram
    fig_stft = go.Figure(data=go.Heatmap(
        z=intensity,
        x=times,
        y=freqs,
        colorscale='Inferno',
        showscale=True
    ))
    
    fig_stft.update_layout(
        title=f"Rolling frequency map ({timeframe} candles)",
        xaxis_title="Time Blocks",
        yaxis_title="Relative Frequency",
        height=500
    )
    st.plotly_chart(fig_stft, use_container_width=True)
    
    # Simple price line for context
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(y=equity_path, mode='lines', name='Price', line=dict(color='#00ff9d')))
    fig_price.update_layout(title="Intraday Price History", height=300)
    st.plotly_chart(fig_price, use_container_width=True)
    
    st.info("💡 **Pro Tip**: In a production setup, this would be hooked up to a real-time WebSocket for tick-by-tick streaming.")
else:
    st.error("We couldn't pull intraday data for that symbol. Check if the market is open or if the symbol is correct.")

