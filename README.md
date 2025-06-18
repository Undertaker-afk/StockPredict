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

# Advanced Stock Prediction Analysis

A comprehensive stock prediction and analysis tool that combines Chronos forecasting with advanced features including regime detection, ensemble methods, and stress testing.

## Features

### Core Prediction Engine
- **Chronos Forecasting**: State-of-the-art time series forecasting using Amazon's Chronos model
- **Technical Analysis**: Traditional technical indicators (RSI, MACD, Bollinger Bands, SMA)
- **Multi-timeframe Support**: Daily, hourly, and 15-minute analysis
- **Real-time Data**: Live market data via yfinance

### Advanced Features

#### 1. Market Regime Detection
- **Hidden Markov Models (HMM)**: Automatic detection of market regimes (bull, bear, sideways)
- **Volatility-based Fallback**: Simplified regime detection when HMM is unavailable
- **Regime-adjusted Signals**: Trading signals that adapt to current market conditions

#### 2. Ensemble Methods
- **Multi-model Combination**: Combines Chronos, technical, and statistical predictions
- **Adaptive Weighting**: User-configurable weights for different models
- **Uncertainty Quantification**: Advanced uncertainty estimation with skewness adjustment

#### 3. Advanced Risk Metrics
- **Tail Risk Analysis**: VaR and CVaR calculations
- **Market Correlation**: Beta, alpha, and correlation with market indices
- **Risk-adjusted Returns**: Sharpe, Sortino, and Calmar ratios
- **Drawdown Analysis**: Maximum drawdown and recovery metrics

#### 4. Stress Testing
- **Scenario Analysis**: Market crash, high volatility, bull market scenarios
- **Interest Rate Shocks**: Impact of rate changes on predictions
- **Custom Scenarios**: User-defined stress test parameters

#### 5. Enhanced Uncertainty Quantification
- **Skewness-aware**: Accounts for non-normal return distributions
- **Adaptive Smoothing**: Reduces prediction drift based on uncertainty
- **Confidence Intervals**: Dynamic confidence levels based on market conditions

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

## Usage

### Basic Analysis
1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
2. Select timeframe (Daily, Hourly, or 15-minute)
3. Choose prediction strategy (Chronos or Technical)
4. Set prediction days and lookback period
5. Click "Analyze Stock"

### Advanced Settings
- **Ensemble Methods**: Enable/disable multi-model combination
- **Regime Detection**: Enable/disable market regime analysis
- **Stress Testing**: Enable/disable scenario analysis
- **Risk-free Rate**: Set annual risk-free rate for calculations
- **Market Index**: Choose correlation index (S&P 500, Dow Jones, NASDAQ, Russell 2000)
- **Ensemble Weights**: Adjust weights for Chronos, Technical, and Statistical models

### Output Sections

#### Daily Analysis
- **Structured Product Metrics**: Market cap, sector, financial ratios
- **Advanced Risk Analysis**: Comprehensive risk metrics with market correlation
- **Market Regime Analysis**: Current regime and transition probabilities
- **Trading Signals**: Advanced signals with confidence levels
- **Stress Test Results**: Scenario analysis outcomes
- **Ensemble Analysis**: Multi-model combination details

#### Hourly/15-minute Analysis
- **Intraday Metrics**: High-frequency volatility and momentum indicators
- **Volume Analysis**: Volume-price trends and momentum
- **Real-time Indicators**: Pre/post market data analysis

## Technical Details

### Regime Detection
- Uses Hidden Markov Models with 3 states (low volatility, normal, high volatility)
- Falls back to volatility-based detection if HMM unavailable
- Regime probabilities influence trading signal thresholds

### Ensemble Methods
- **Chronos**: Primary deep learning model (60% default weight)
- **Technical**: Traditional indicators with mean reversion (20% default weight)
- **Statistical**: ARIMA-like models with momentum (20% default weight)

### Stress Testing Scenarios
- **Market Crash**: 3x volatility, -15% return shock
- **High Volatility**: 2x volatility, -5% return shock
- **Low Volatility**: 0.5x volatility, +2% return shock
- **Bull Market**: 1.2x volatility, +10% return shock
- **Interest Rate Shock**: 1.5x volatility, -8% return shock

### Uncertainty Quantification
- Skewness-adjusted confidence intervals
- Adaptive smoothing based on prediction uncertainty
- Time-varying volatility modeling

## Dependencies

### Core
- `torch>=2.1.2`: PyTorch for deep learning
- `chronos-forecasting>=1.0.0`: Amazon's Chronos model
- `yfinance>=0.2.0`: Yahoo Finance data
- `gradio>=4.0.0`: Web interface

### Advanced Features
- `hmmlearn>=0.3.0`: Hidden Markov Models for regime detection
- `scipy>=1.10.0`: Scientific computing and statistics
- `scikit-learn>=1.0.0`: Machine learning utilities
- `plotly>=5.0.0`: Interactive visualizations

## Limitations

1. **Market Hours**: Intraday data (hourly/15-minute) only available during market hours
2. **Data Quality**: Dependent on yfinance data availability and quality
3. **Model Complexity**: Advanced features may increase computation time
4. **GPU Requirements**: Chronos model requires CUDA-capable GPU for optimal performance

## Disclaimer

This tool is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct thorough research and consider consulting with financial professionals before making investment decisions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Practical Example: Creating a 6-Month 8% Yield Structured Product

### Scenario
A bank needs to create a structured product that offers an 8% yield over 6 months while maintaining profitability for the institution.

### Step-by-Step Implementation

1. **Initial Stock Screening**
   - Use the application to analyze stocks with:
     - High liquidity (for easy hedging)
     - Stable volatility (for predictable risk)
     - Strong technical indicators
     - Positive long-term trends
   - Recommended stocks: AAPL, MSFT, GOOGL (high liquidity, stable volatility)

2. **Product Structure Design**
   - Use the 6-month prediction horizon
   - Analyze historical volatility for barrier setting
   - Set participation rate based on risk metrics
   - Structure: Reverse Convertible with 8% coupon

3. **Risk Analysis**
   - Use the application's risk metrics:
     - Check maximum drawdown (should be < 15% for 6 months)
     - Verify liquidity scores (should be > 80%)
     - Analyze Sharpe ratio (should be > 1.5)

4. **Business Case Example**

   **Product Parameters:**
   - Notional Amount: $1,000,000
   - Term: 6 months
   - Coupon: 8% p.a. (4% for 6 months)
   - Underlying: AAPL
   - Barrier: 85% of initial price
   - Participation: 100%

   **Revenue Structure:**
   - Client receives: 8% p.a. (4% for 6 months)
   - Bank's hedging cost: ~5% p.a.
   - Bank's profit margin: ~3% p.a.
   - Total client payout: $40,000 (4% of $1M)
   - Bank's profit: $15,000 (1.5% of $1M)

5. **Implementation Steps**
   - Use the application's extended prediction horizon (180 days)
   - Set technical indicators to monitor barrier risk
   - Implement dynamic delta hedging based on predictions
   - Monitor risk metrics daily using the application

6. **Risk Management**
   - Use the application's volatility predictions for dynamic hedging
   - Monitor technical indicators for early warning signals
   - Set up automated alerts for barrier proximity
   - Regular rebalancing based on prediction updates

### Key Success Factors
- Regular monitoring of prediction accuracy
- Dynamic adjustment of hedging strategy
- Clear communication of product risks to clients
- Proper documentation of all assumptions and methodologies

This example demonstrates how the application can be used to create profitable structured products while managing risk effectively. The bank can use this framework to create similar products with different underlying assets, terms, and yield targets.
