---
title: Advanced Stock Prediction Analysis with Amazon Chronos
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
short_description: Advanced AI-powered stock prediction system with Amazon Chronos foundation model
tags:
  - mcp-server-track
  - finance
  - machine-learning
  - time-series
  - stock-prediction
  - chronos
---

# 🚀 Advanced Stock Prediction Analysis with Amazon Chronos

A cutting-edge AI-powered stock prediction and analysis system with **580M+ parameters**, designed to analyze and predict stock prices across multiple timeframes. Equipped with **Amazon's Chronos foundation model** and **advanced ensemble methods**, it excels in both short-term trading and long-term investment analysis.

## 🌟 Key Features

### Market Status Monitoring
- **Real-Time Market Status**: Check if markets are open or closed with a simple click
- **Multi-Market Support**: Monitor US Stocks, European Markets, Asian Markets, Forex, Crypto, Futures, and Commodities
- **Timezone-Aware**: Accurate status based on each market's local timezone
- **Trading Hours**: Detailed information about market hours and next trading days
- **24/7 Markets**: Support for continuous trading markets like Forex and Crypto
- **User-Friendly Interface**: Simple dropdown menu and click-to-check functionality

### Core Prediction Engine
- **Amazon Chronos Integration**: Uses the state-of-the-art Chronos T5 foundation model for probabilistic time series forecasting
- **Multi-Timeframe Analysis**: Support for daily, hourly, and 15-minute timeframes
- **Advanced Ensemble Methods**: Combines multiple algorithms including Random Forest, Gradient Boosting, SVR, and Neural Networks

### Enhanced Covariate Data
- **Market Indices**: S&P 500, Dow Jones, NASDAQ, VIX, Treasury yields
- **Sector ETFs**: Financial, Technology, Energy, Healthcare, and more
- **Commodities**: Gold, Silver, Oil, Natural Gas, Corn, Soybeans
- **Currencies**: EUR/USD, GBP/USD, JPY/USD, CHF/USD, CAD/USD
- **Economic Indicators**: Inflation proxies, volatility indices, dollar strength

### Advanced Uncertainty Calculations
- **Multiple Uncertainty Methods**:
  - Basic quantile-based uncertainty
  - Skewness-adjusted uncertainty
  - Volatility-scaled uncertainty
  - Market condition adjusted uncertainty
  - Time-decay uncertainty
  - Ensemble uncertainty (combines all methods)
- **Regime-Aware Uncertainty**: Adjusts uncertainty based on market regime detection
- **Confidence Intervals**: 95% confidence bands with multiple calculation methods

### Enhanced Volume Prediction
- **Price-Volume Relationship Modeling**: Analyzes the relationship between price movements and volume
- **Volume Momentum**: Incorporates volume momentum and trends
- **Market Condition Adjustments**: Adjusts volume predictions based on market volatility
- **Uncertainty Quantification**: Provides volume prediction uncertainty estimates

### Sentiment Analysis
- **News Sentiment Scoring**: Analyzes news articles for sentiment polarity
- **Confidence Levels**: Provides confidence scores for sentiment analysis
- **Real-time Integration**: Incorporates sentiment data into prediction models

### Market Regime Detection
- **Hidden Markov Models**: Detects bull, bear, and sideways market regimes
- **Volatility Clustering**: Identifies periods of high and low volatility
- **Regime-Aware Predictions**: Adjusts predictions based on current market regime

### Advanced Algorithms
- **Multi-Algorithm Ensemble**:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Ridge Regression
  - Lasso Regression
  - Support Vector Regression (SVR)
  - Multi-Layer Perceptron (MLP)
- **Time Series Cross-Validation**: Uses expanding window validation for robust model evaluation
- **Weighted Ensemble**: Combines predictions using uncertainty-weighted averaging

### Financial Smoothing
- **Multiple Smoothing Methods**:
  - Exponential smoothing (trend following)
  - Moving average (noise reduction)
  - Kalman filter (adaptive smoothing)
  - Savitzky-Golay (preserves peaks/valleys)
  - Double exponential (trend + level)
  - Triple exponential (complex patterns)
  - Adaptive smoothing (volatility-based)

## 📊 Technical Indicators

### Price-Based Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator with regime-adjusted thresholds
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility indicator with position analysis
- **Moving Averages**: SMA 20, SMA 50 with crossover analysis

### Volume-Based Indicators
- **Volume-Price Trend**: Analyzes the relationship between volume and price movements
- **Volume Momentum**: Tracks volume changes over time
- **Volume Volatility**: Measures volume stability
- **Volume Ratio**: Compares current volume to historical averages

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Value at Risk (VaR)**: Maximum expected loss at given confidence level
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Beta**: Market correlation measure
- **Volatility**: Historical and implied volatility measures

## 🛠️ Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Key Dependencies**:
- `chronos-forecasting>=1.0.0`: Amazon's Chronos foundation model
- `torch>=2.1.2`: PyTorch for deep learning
- `yfinance>=0.2.0`: Yahoo Finance data
- `scikit-learn>=1.3.0`: Machine learning algorithms
- `plotly>=5.0.0`: Interactive visualizations
- `gradio>=4.0.0`: Web interface
- `textblob>=0.17.1`: Sentiment analysis
- `arch>=6.2.0`: GARCH modeling
- `hmmlearn>=0.3.0`: Hidden Markov Models

## 🚀 Usage

### Web Interface
```bash
python app.py
```

The application provides a comprehensive web interface with three main tabs:

1. **Daily Analysis**: Long-term investment analysis (up to 365 days)
2. **Hourly Analysis**: Medium-term trading analysis (up to 7 days)
3. **15-Minute Analysis**: Short-term scalping analysis (up to 3 days)

### Market Status Check
The application includes a simple market status monitoring feature:

1. **Quick Market Status Check**: Located at the top of the interface
2. **Market Selection**: Dropdown menu with all supported markets:
   - US Stock Market (NYSE, NASDAQ, AMEX)
   - European Markets (London, Frankfurt, Paris)
   - Asian Markets (Tokyo, Hong Kong, Shanghai)
   - Forex Market (24/7 Global Currency Exchange)
   - Cryptocurrency Market (24/7 Bitcoin, Ethereum, Altcoins)
   - US Futures Market (24/7 CME, ICE, CBOT)
   - Commodities Market (24/7 Gold, Silver, Oil, Natural Gas)
3. **One-Click Check**: Click "Check Market Status" to get real-time information
4. **Detailed Information**: Shows current status, trading hours, next trading day, and time until open/close

**Example Output:**
```
🟢 US Stock Market Status: OPEN

Current Status: US Stock Market is currently open

Market Details:
- Type: Stocks
- Symbol: ^GSPC
- Current Time: 14:30:00 EDT
- Last Updated: 2024-01-15 14:30:00 EDT

Trading Information:
- Next Trading Day: 2024-01-16
- Time Until Open: N/A (Market is open)
- Time Until Close: 1h 30m

Market Description: NYSE, NASDAQ, AMEX
```

### Advanced Settings
- **Ensemble Methods**: Enable/disable multi-algorithm ensemble
- **Regime Detection**: Enable/disable market regime detection
- **Stress Testing**: Enable/disable scenario analysis
- **Enhanced Covariate Data**: Include market indices, sectors, commodities
- **Sentiment Analysis**: Include news sentiment analysis
- **Smoothing**: Choose from multiple smoothing algorithms

### Ensemble Weights
Configure the weights for different prediction methods:
- **Chronos Weight**: Weight for Amazon Chronos predictions
- **Technical Weight**: Weight for technical analysis
- **Statistical Weight**: Weight for statistical models

## 📈 Prediction Features

### Enhanced Uncertainty Quantification
The system provides multiple uncertainty calculation methods:

1. **Basic Uncertainty**: Standard quantile-based uncertainty
2. **Skewness-Adjusted**: Accounts for asymmetric return distributions
3. **Volatility-Scaled**: Scales uncertainty based on historical volatility
4. **Market-Adjusted**: Adjusts uncertainty based on VIX and market conditions
5. **Time-Decay**: Uncertainty increases with prediction horizon
6. **Ensemble Uncertainty**: Combines all methods for robust estimates

### Volume Prediction Improvements
- **Price-Volume Relationship**: Models the relationship between price movements and volume
- **Momentum Effects**: Incorporates volume momentum and trends
- **Market Condition Adjustments**: Adjusts predictions based on market volatility
- **Uncertainty Quantification**: Provides confidence intervals for volume predictions

### Covariate Integration
The system automatically collects and integrates:
- **Market Indices**: S&P 500, Dow Jones, NASDAQ, VIX
- **Sector Performance**: Financial, Technology, Energy, Healthcare ETFs
- **Economic Indicators**: Treasury yields, dollar index, commodity prices
- **Global Markets**: International indices and currencies

## 🔬 Advanced Features

### Regime Detection
Uses Hidden Markov Models to detect market regimes:
- **Bull Market**: High returns, low volatility
- **Bear Market**: Low returns, high volatility  
- **Sideways Market**: Low returns, low volatility

### Stress Testing
Performs scenario analysis under various market conditions:
- **Market Crash**: -20% market decline
- **Volatility Spike**: 50% increase in VIX
- **Interest Rate Shock**: 100 basis point rate increase
- **Sector Rotation**: Major sector performance shifts

### Sentiment Analysis
- **News Sentiment**: Analyzes recent news articles for sentiment
- **Confidence Scoring**: Provides confidence levels for sentiment analysis
- **Integration**: Incorporates sentiment into prediction models

## 📊 Output Metrics

### Trading Signals
- **RSI Signals**: Oversold/Overbought with confidence levels
- **MACD Signals**: Buy/Sell with strength indicators
- **Bollinger Bands**: Position within bands with breakout signals
- **SMA Signals**: Trend following with crossover analysis

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **VaR**: Value at Risk at 95% confidence
- **Maximum Drawdown**: Largest historical decline
- **Beta**: Market correlation coefficient
- **Volatility**: Historical and implied volatility

### Enhanced Features
- **Covariate Data Usage**: Indicates which external data was used
- **Sentiment Analysis**: News sentiment scores and confidence
- **Advanced Uncertainty Methods**: List of uncertainty calculation methods used
- **Regime-Aware Uncertainty**: Whether regime detection was applied
- **Enhanced Volume Prediction**: Whether advanced volume modeling was used

## 🎯 Use Cases

### Long-Term Investment (Daily Analysis)
- Portfolio management and asset allocation
- Strategic investment decisions
- Risk management and hedging
- Sector rotation strategies

### Medium-Term Trading (Hourly Analysis)
- Swing trading strategies
- Position sizing and timing
- Intraday volatility analysis
- Momentum-based trading

### Short-Term Trading (15-Minute Analysis)
- Scalping strategies
- High-frequency trading
- Micro-pattern recognition
- Ultra-short-term momentum

## 🔧 Configuration

### Model Parameters
- **Chronos Model**: `amazon/chronos-t5-large` (default)
- **Context Window**: 64 time steps
- **Prediction Length**: Configurable up to model limits
- **Quantile Levels**: [0.1, 0.5, 0.9] for uncertainty estimation

### Data Sources
- **Primary**: Yahoo Finance (yfinance)
- **Covariates**: Market indices, ETFs, commodities, currencies
- **Sentiment**: News articles via yfinance
- **Economic Data**: Treasury yields, VIX, dollar index

## 📝 Notes

- **Market Hours**: Hourly and 15-minute data only available during market hours
- **Data Limitations**: Yahoo Finance has specific limits for intraday data
- **Model Performance**: Chronos performs best with sufficient historical data
- **Uncertainty**: All predictions include comprehensive uncertainty estimates
- **Ensemble Weights**: Should sum to 1.0 for optimal performance

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:
- Additional covariate data sources
- New uncertainty calculation methods
- Advanced sentiment analysis techniques
- Custom technical indicators
- Alternative ensemble methods

## 📄 License

This project is licensed under the Apache-2.0 License.

## 🙏 Acknowledgments

- **Amazon Chronos**: Foundation model for time series forecasting
- **Yahoo Finance**: Market data provider
- **Gradio**: Web interface framework
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
