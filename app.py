import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz

from data_fetcher import fetch_historical_data, get_symbol_map
from technical_indicators import add_indicators
from prediction_model import train_prediction_model, predict_next_move, calculate_price_levels, generate_trading_signal
from visualize import plot_market_data, plot_prediction_confidence, plot_trading_signal, plot_market_sessions
from market_sessions import get_market_session_info
from image_analyzer import process_uploaded_image, display_chart_analysis
from news_alerts import display_news_alert

# Set page configuration
st.set_page_config(
    page_title="Forex Prediction App",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Forex Prediction App")
st.markdown("""
    This application visualizes market data and predicts price movements for selected financial instruments.
    Choose an instrument and time period to see historical data, technical indicators, and price predictions.
""")

# Sidebar with instrument selection and time period
st.sidebar.header("Settings")

# Instrument selection
symbol_map = get_symbol_map()
instrument = st.sidebar.selectbox(
    "Select Instrument",
    list(symbol_map.keys()),
    index=0
)

# Time period selection
period_options = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y"
}
period = st.sidebar.selectbox(
    "Select Time Period",
    list(period_options.keys()),
    index=2
)

# Interval selection
interval_options = {
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "1 Day": "1d"
}
interval = st.sidebar.selectbox(
    "Select Interval",
    list(interval_options.keys()),
    index=4 if period_options[period] in ["3mo", "6mo", "1y"] else 2
)

# Technical indicators selection
indicators = st.sidebar.multiselect(
    "Select Technical Indicators",
    ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
    default=["SMA", "RSI"]
)

# Refresh rate
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 
                                        min_value=30, 
                                        max_value=300, 
                                        value=60, 
                                        step=30)

# Prediction settings
st.sidebar.header("Prediction Settings")
prediction_horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    ["Next Period", "Next Day", "Next Week"],
    index=0
)

# Trading Signal Settings
st.sidebar.header("Trading Signal Settings")

show_trading_signals = st.sidebar.checkbox("Show Trading Signals", value=True)

# Risk-Reward Ratio Slider
risk_reward_ratio = st.sidebar.slider(
    "Risk-Reward Ratio", 
    min_value=1.0, 
    max_value=5.0, 
    value=2.0, 
    step=0.1
)

# Prediction Timeframe
prediction_timeframe = st.sidebar.selectbox(
    "Prediction Timeframe",
    [interval_options[interval], "1h", "4h", "1d", "1w"],
    index=0
)

# Set up the main interface
col1, col2 = st.columns([2, 1])

# Progress bar for loading data
with st.spinner(f"Fetching data for {instrument}..."):
    # Get the actual symbol based on the user selection
    symbol = symbol_map[instrument]
    
    # Fetch historical data
    df = fetch_historical_data(
        symbol=symbol,
        period=period_options[period],
        interval=interval_options[interval]
    )
    
    if df is not None and not df.empty:
        # Add technical indicators
        df = add_indicators(df, indicators)
        
        # Display basic statistics
        with col2:
            st.subheader(f"{instrument} Statistics")
            
            # Last update time
            last_update = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(f"**Last Updated:** {last_update}")
            
            # Latest prices
            latest = df.iloc[-1]
            current_price = latest['Close']
            
            # Calculate price change
            prev_price = df.iloc[-2]['Close'] if len(df) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            price_color = "green" if price_change >= 0 else "red"
            st.markdown(f"**Current Price:** <span style='color:{price_color}'>${current_price:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Change:** <span style='color:{price_color}'>${price_change:.2f} ({price_change_pct:.2f}%)</span>", unsafe_allow_html=True)
            
            # Display high/low
            st.markdown(f"**Day High:** ${df['High'].iloc[-1]:.2f}")
            st.markdown(f"**Day Low:** ${df['Low'].iloc[-1]:.2f}")
            
            # Volume if available
            if 'Volume' in df.columns:
                st.markdown(f"**Volume:** {df['Volume'].iloc[-1]:,.0f}")
            
            # Display key indicators
            st.subheader("Technical Indicators")
            
            indicator_values = {}
            
            if "SMA" in indicators and "SMA_20" in df.columns:
                sma_value = df["SMA_20"].iloc[-1]
                indicator_values["SMA (20)"] = f"${sma_value:.2f}"
                
            if "EMA" in indicators and "EMA_14" in df.columns:
                ema_value = df["EMA_14"].iloc[-1]
                indicator_values["EMA (14)"] = f"${ema_value:.2f}"
                
            if "RSI" in indicators and "RSI_14" in df.columns:
                rsi_value = df["RSI_14"].iloc[-1]
                rsi_color = "green" if rsi_value < 30 else "red" if rsi_value > 70 else "black"
                indicator_values["RSI (14)"] = f"<span style='color:{rsi_color}'>{rsi_value:.2f}</span>"
                
            if "MACD" in indicators and "MACD" in df.columns:
                macd_value = df["MACD"].iloc[-1]
                macd_signal = df["MACD_Signal"].iloc[-1]
                macd_hist = df["MACD_Hist"].iloc[-1]
                indicator_values["MACD"] = f"{macd_value:.2f}"
                indicator_values["MACD Signal"] = f"{macd_signal:.2f}"
                indicator_values["MACD Hist"] = f"{macd_hist:.2f}"
                
            if "Bollinger Bands" in indicators and "BB_Upper" in df.columns:
                bb_upper = df["BB_Upper"].iloc[-1]
                bb_middle = df["BB_Middle"].iloc[-1]
                bb_lower = df["BB_Lower"].iloc[-1]
                indicator_values["BB Upper"] = f"${bb_upper:.2f}"
                indicator_values["BB Middle"] = f"${bb_middle:.2f}"
                indicator_values["BB Lower"] = f"${bb_lower:.2f}"
            
            for indicator, value in indicator_values.items():
                st.markdown(f"**{indicator}:** {value}", unsafe_allow_html=True)
            
        # Visualize the market data
        with col1:
            st.subheader(f"{instrument} Market Data")
            fig = plot_market_data(df, instrument, indicators)
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction section
        st.subheader(f"{instrument} Price Movement Prediction")
        
        # Train prediction model and make prediction
        X, y, model, X_predict, prediction, confidence = train_prediction_model(df, horizon=prediction_horizon)
        
        # Display prediction
        col3, col4 = st.columns([2, 1])
        
        with col3:
            # Display prediction confidence chart
            fig_confidence = plot_prediction_confidence(prediction, confidence)
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        with col4:
            st.subheader("Prediction Summary")
            
            # Determine direction and emoji
            direction = "Up" if prediction > 0.5 else "Down"
            emoji = "📈" if prediction > 0.5 else "📉"
            
            # Display prediction with color
            pred_color = "green" if prediction > 0.5 else "red"
            st.markdown(f"### Predicted Direction: <span style='color:{pred_color}'>{direction} {emoji}</span>", unsafe_allow_html=True)
            
            # Display confidence
            confidence_str = f"{confidence:.2%}"
            st.markdown(f"### Confidence: {confidence_str}")
            
            # Display confidence level
            confidence_level = "High" if confidence >= 0.75 else "Medium" if confidence >= 0.6 else "Low"
            st.markdown(f"### Confidence Level: {confidence_level}")
            
            # Prediction timestamp
            pred_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(f"*Prediction made at: {pred_time}*")
            
            # Notes section
            st.info("""
            **Note:** This prediction is based on historical data patterns and technical indicators. 
            Market conditions can change rapidly due to unforeseen events. Always conduct your own research 
            before making trading decisions.
            """)
            
        # Trading Signals Section
        if show_trading_signals:
            st.subheader(f"{instrument} Trading Signals")
            
            # Generate trading signal based on prediction and selected timeframe
            trading_signal = generate_trading_signal(
                df=df, 
                timeframe=prediction_timeframe,
                prediction_prob=prediction,
                confidence=confidence
            )
            
            # Calculate price levels with selected risk-reward ratio
            price_levels = calculate_price_levels(
                df=df,
                prediction_prob=prediction,
                confidence=confidence,
                risk_reward_ratio=risk_reward_ratio
            )
            
            # Update trading signal with custom risk-reward ratio
            trading_signal['risk_reward_ratio'] = risk_reward_ratio
            trading_signal['stop_loss_price'] = price_levels['stop_loss_price']
            trading_signal['take_profit_price'] = price_levels['take_profit_price']
            
            # Display trading signal details
            col5, col6 = st.columns([1, 1])
            
            with col5:
                # Display signal visualization
                fig_signal = plot_trading_signal(trading_signal)
                st.plotly_chart(fig_signal, use_container_width=True)
            
            with col6:
                st.subheader("Signal Details")
                
                # Signal type with color
                signal_type = trading_signal["signal_type"]
                signal_color = "green" if signal_type == "BUY" else "red"
                st.markdown(f"**Signal Type:** <span style='color:{signal_color}'>{signal_type}</span>", unsafe_allow_html=True)
                
                # Price levels
                st.markdown(f"**Entry Price:** ${trading_signal['entry_price']:.2f}")
                st.markdown(f"**Stop Loss:** ${trading_signal['stop_loss_price']:.2f}")
                st.markdown(f"**Take Profit:** ${trading_signal['take_profit_price']:.2f}")
                
                # Calculate pip values and risk
                entry_price = trading_signal['entry_price']
                stop_loss = trading_signal['stop_loss_price']
                take_profit = trading_signal['take_profit_price']
                
                # Calculate risk and reward in pips/points
                if instrument == "Gold (XAU/USD)":
                    # Gold is quoted in USD per troy ounce
                    risk_points = abs(entry_price - stop_loss)
                    reward_points = abs(entry_price - take_profit)
                    point_label = "points"
                else:
                    # Forex and indices use pips/points
                    risk_points = abs(entry_price - stop_loss) * 10000 if "USD" in instrument else abs(entry_price - stop_loss) * 100
                    reward_points = abs(entry_price - take_profit) * 10000 if "USD" in instrument else abs(entry_price - take_profit) * 100
                    point_label = "pips" if "USD" in instrument else "points"
                
                st.markdown(f"**Risk:** {risk_points:.1f} {point_label}")
                st.markdown(f"**Reward:** {reward_points:.1f} {point_label}")
                st.markdown(f"**Risk-Reward Ratio:** 1:{risk_reward_ratio:.1f}")
                
                # Display timeframe
                st.markdown(f"**Timeframe:** {trading_signal['timeframe']}")
                
                # Display confidence
                confidence_str = f"{trading_signal['confidence']:.2%}"
                st.markdown(f"**Signal Confidence:** {confidence_str}")
                
                # Signal timestamp
                signal_time = trading_signal['signal_time'].strftime('%Y-%m-%d %H:%M:%S')
                st.markdown(f"*Signal generated at: {signal_time}*")
            
            # Display market chart with signal
            st.subheader(f"{instrument} Chart with Trading Signal")
            
            # Get chart with signals
            fig_with_signals = plot_market_data(df, instrument, indicators, signal=trading_signal)
            st.plotly_chart(fig_with_signals, use_container_width=True)
            
            # Trading signal disclaimer
            st.warning("""
            **Disclaimer:** Trading signals are generated based on historical data analysis and 
            technical indicators. They should not be considered as financial advice. 
            Always use proper risk management and consider your financial situation before 
            making trading decisions.
            """)
            
        # Market Sessions Information Section
        st.subheader("Market Sessions & Liquidity")
        
        # Get market session information
        market_info = get_market_session_info()
        
        # Display market sessions visual
        fig_sessions = plot_market_sessions()
        st.plotly_chart(fig_sessions, use_container_width=True)
        
        # Display current session information
        current_session = market_info["current_session"]
        
        # Create columns for session information
        col_session1, col_session2 = st.columns(2)
        
        with col_session1:
            st.subheader("Current Session")
            
            # Determine liquidity color
            liquidity_color = "green" if current_session["liquidity"] in ["High", "Very High"] else "orange" if current_session["liquidity"] == "Medium" else "red"
            
            # Display current session info
            st.markdown(f"**Active Session:** {current_session['session']}")
            st.markdown(f"**Liquidity:** <span style='color:{liquidity_color}'>{current_session['liquidity']}</span>", unsafe_allow_html=True)
            
            if current_session["is_active"] and current_session["time_remaining"]:
                st.markdown(f"**Time Remaining:** {current_session['time_remaining']}")
            
            st.markdown(f"**Session Info:** {current_session['description']}")
            
            # Add an alert for high liquidity periods
            if current_session["liquidity"] in ["High", "Very High"]:
                st.success(f"🔔 **ALERT:** This is a high liquidity period, optimal for trading!")
            
        with col_session2:
            st.subheader("Upcoming Session")
            
            # Display upcoming session info
            upcoming = market_info["upcoming_session"]
            
            st.markdown(f"**Next Session:** {upcoming['session']}")
            st.markdown(f"**Starting In:** {upcoming['time_until_start']}")
            st.markdown(f"**Expected Liquidity:** {upcoming['liquidity']}")
            st.markdown(f"**Session Info:** {upcoming['description']}")
        
        # Display high-impact economic events
        st.subheader("High-Impact Economic Events Today")
        
        high_impact_events = market_info["high_impact_events"]
        
        if high_impact_events:
            events_df = pd.DataFrame(high_impact_events)
            st.table(events_df)
            
            # Add alert for high-impact events
            st.info("🔔 **ALERT:** High-impact economic events can cause significant market volatility. Be cautious when trading during these times.")
    else:
        st.error(f"Failed to fetch data for {instrument}. Please try another instrument or time period.")

# News Alerts and Fundamental Analysis Section
st.subheader("News Alerts & Trade Management")
st.markdown("""
    Monitor economic events that could impact your trading decisions. 
    Get alerts about high-impact news and recommendations on whether to close your trades.
""")

# Display news alerts for the selected instrument
# Safely handle trading signal parameter
try:
    if show_trading_signals and 'trading_signal' in locals():
        display_news_alert(instrument, trading_signal)
    else:
        display_news_alert(instrument)
except Exception as e:
    st.error(f"Error displaying news alerts: {str(e)}")
    # Fallback to basic display without trading signal
    display_news_alert(instrument)

# Image Upload and Analysis Section
st.subheader("Forex Chart Image Analysis")
st.markdown("""
    Upload a forex or stock chart image for AI-powered technical analysis.
    The AI will analyze the chart and provide predictions based on patterns and indicators visible in the image.
""")

uploaded_file = st.file_uploader("Upload a chart image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Process and display the uploaded image
    image, analysis_result = process_uploaded_image(uploaded_file)
    
    # Create two columns for image and analysis
    img_col, analysis_col = st.columns([1, 1])
    
    with img_col:
        st.image(image, caption="Uploaded Chart Image", use_column_width=True)
    
    with analysis_col:
        # Display the analysis results
        display_chart_analysis(analysis_result)

# Auto-refresh logic
if auto_refresh:
    st.markdown(f"*Auto-refreshing every {refresh_interval} seconds*")
    time.sleep(refresh_interval)
    st.rerun()
