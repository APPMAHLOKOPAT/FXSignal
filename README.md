# Forex Prediction App

A comprehensive Streamlit-based forex prediction application for trading instruments including Gold (XAU/USD), NAS100, US30, and GER30. This app visualizes market data, predicts price movements, and provides trading signals with specific entry, stop loss, and take profit levels.

## Features

### Real-time Market Data
- Interactive price charts with candlestick patterns
- Multiple technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Customizable time periods and intervals

### Price Prediction
- Machine learning-based price movement predictions
- Confidence metrics with visual representation
- Support for different prediction horizons (next period, next day, next week)

### Trading Signals
- Automated trading signals with entry, stop loss, and take profit levels
- Adjustable risk-reward ratio settings
- Trading signal visualization

### Market Session Tracking
- Real-time monitoring of forex market sessions (Sydney, Tokyo, London, New York)
- High liquidity period alerts when major sessions overlap
- Visual timeline of market sessions throughout the day

### Economic Calendar & News Alerts
- Economic events calendar with impact assessment on selected instruments
- Volatility forecasts based on upcoming economic events
- Trade management recommendations for open positions
- News sentiment analysis with AI-powered market impact assessment

### Chart Image Analysis
- Upload chart images for AI-powered technical analysis
- Pattern recognition in uploaded charts
- Support and resistance level identification
- Trading recommendations based on visual chart patterns

## Project Structure

- **app.py**: Main application file that creates the Streamlit interface
- **data_fetcher.py**: Handles fetching market data from Yahoo Finance
- **technical_indicators.py**: Calculates technical indicators like SMA, EMA, RSI, etc.
- **prediction_model.py**: Contains the machine learning model for price predictions
- **visualize.py**: Creates interactive visualizations and charts
- **market_sessions.py**: Tracks forex market sessions and liquidity periods
- **news_alerts.py**: Provides economic calendar and trade management recommendations
- **image_analyzer.py**: Analyzes uploaded chart images using OpenAI's vision capabilities
- **.streamlit/config.toml**: Streamlit configuration file

## Requirements

- Python 3.11+
- Key packages:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - plotly
  - yfinance
  - openai
  - pytz

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forex-prediction-app.git
cd forex-prediction-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Select an instrument from the sidebar
2. Choose time period and interval
3. Select technical indicators to display
4. Adjust prediction horizon and trading signal settings
5. Monitor market session information for optimal trading times
6. Check economic calendar for upcoming high-impact events
7. Upload chart images for AI analysis when needed

## Deployment

The app is configured for easy deployment on Streamlit Sharing, Heroku, or any other platform that supports Python web applications.

## License

[MIT License](LICENSE)