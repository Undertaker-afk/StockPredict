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

A comprehensive stock analysis and prediction tool built with Gradio, featuring multiple prediction strategies and technical analysis indicators. The application is particularly suited for structured financial product creation and analysis.

## Features

- **Multiple Prediction Strategies**:
  - Chronos ML-based prediction
  - Technical analysis-based prediction

- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Simple Moving Averages (20, 50, and 200-day)

- **Trading Signals**:
  - Buy/Sell recommendations based on multiple indicators
  - Overall trading signal combining all indicators
  - Confidence intervals for predictions

- **Interactive Visualizations**:
  - Price prediction with confidence intervals
  - Technical indicators overlay
  - Volume analysis
  - Historical price trends

- **Structured Product Analysis**:
  - Extended prediction horizons (up to 1 year)
  - Historical analysis up to 10 years
  - Comprehensive risk metrics
  - Sector and industry analysis
  - Liquidity assessment

## Structured Product Features

### Extended Time Horizons
- Prediction window up to 365 days
- Historical data analysis up to 10 years
- Long-term trend analysis
- Extended technical indicators

### Risk Analysis
- Annualized volatility
- Maximum drawdown analysis
- Current drawdown tracking
- Sharpe and Sortino ratios
- Risk-adjusted return metrics

### Product Metrics
- Market capitalization
- Sector and industry classification
- Dividend yield analysis
- Volume metrics
- Liquidity scoring

### Sector Analysis
- Market cap ranking (Large/Mid/Small)
- Sector exposure
- Industry classification
- Liquidity assessment

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
   - Number of days to predict (up to 365 days)
   - Historical lookback period (up to 10 years)
   - Prediction strategy (Chronos or Technical)

4. Click "Analyze Stock" to get:
   - Price predictions and trading signals
   - Structured product metrics
   - Risk analysis
   - Sector analysis

## Using for Structured Products

### Initial Screening
1. Use extended lookback period (up to 10 years) for long-term performance analysis
2. Look for stocks with stable volatility and good risk-adjusted returns
3. Check liquidity scores for trading feasibility

### Risk Assessment
1. Review risk metrics to match client risk profile
2. Analyze maximum drawdowns for worst-case scenarios
3. Compare risk-adjusted returns using Sharpe and Sortino ratios

### Product Structuring
1. Use prediction horizon (up to 1 year) for product maturity design
2. Consider dividend yields for income-generating products
3. Use sector analysis for proper diversification

### Portfolio Construction
1. Analyze multiple stocks for diversified bundles
2. Use sector metrics to avoid overexposure
3. Consider market cap rankings for appropriate sizing

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
