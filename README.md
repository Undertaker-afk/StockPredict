---
title: Stock Predictions
emoji: 🐢
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
short_description: Use Amazon Chronos To Predict Stock Prices
tags:
  - mcp-server-track
---

# Stock Predictions with Chronos

A powerful stock prediction application that uses the Chronos model and technical analysis to predict stock prices across different timeframes.

## Features

- Multiple timeframe analysis (Daily, Hourly, 15-minute)
- Integration with Chronos AI model for predictions
- Technical analysis with multiple indicators
- Real-time market data using Yahoo Finance
- Beautiful interactive visualizations using Plotly
- Risk analysis and sector metrics
- Trading signals generation

## Requirements

- Python 3.8+
- PyTorch
- Gradio
- yfinance
- pandas
- numpy
- plotly
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-predictions.git
cd stock-predictions
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python app.py
```

The application will start and provide a local URL (usually http://127.0.0.1:7860) where you can access the interface.

## Features

- **Daily Analysis**: Long-term predictions and analysis
- **Hourly Analysis**: Medium-term predictions with 30-day lookback
- **15-Minute Analysis**: Short-term predictions with 5-day lookback
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMAs
- **Risk Metrics**: Volatility, Drawdown, Sharpe Ratio
- **Sector Analysis**: Market cap, industry classification, liquidity metrics

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
