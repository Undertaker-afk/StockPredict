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
---

# Stock Analysis and Prediction Demo

A comprehensive stock analysis and prediction tool built with Gradio, featuring multiple prediction strategies and technical analysis indicators.

## Features

- **Multiple Prediction Strategies**:
  - Chronos ML-based prediction
  - Technical analysis-based prediction

- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Simple Moving Averages (20 and 50-day)

- **Trading Signals**:
  - Buy/Sell recommendations based on multiple indicators
  - Overall trading signal combining all indicators
  - Confidence intervals for predictions

- **Interactive Visualizations**:
  - Price prediction with confidence intervals
  - Technical indicators overlay
  - Volume analysis
  - Historical price trends

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Gradio demo:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT) and select your desired parameters:
   - Timeframe (1d, 1h, 15m)
   - Number of days to predict
   - Prediction strategy (Chronos or Technical)

4. Click "Analyze Stock" to get predictions and trading signals

## Prediction Strategies

### Chronos Strategy
Uses Amazon's Chronos model for ML-based price prediction. This strategy:
- Analyzes historical price patterns
- Generates probabilistic forecasts
- Provides confidence intervals

### Technical Strategy
Uses traditional technical analysis indicators to generate predictions:
- RSI for overbought/oversold conditions
- MACD for trend direction
- Bollinger Bands for volatility
- Moving Averages for trend confirmation

## Trading Signals

The demo provides trading signals based on multiple technical indicators:
- RSI: Oversold (<30), Overbought (>70), Neutral
- MACD: Buy (MACD > Signal), Sell (MACD < Signal)
- Bollinger Bands: Buy (price < lower band), Sell (price > upper band)
- SMA: Buy (20-day > 50-day), Sell (20-day < 50-day)

An overall trading signal is calculated by combining all individual signals.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
