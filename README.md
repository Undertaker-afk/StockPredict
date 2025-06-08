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

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Stock Price Prediction with Amazon Chronos

A neural network application that uses Amazon's Chronos model for time series forecasting to predict stock prices.

## Features

- Real-time stock price predictions using Amazon Chronos
- Interactive visualization of predictions with confidence intervals
- Support for multiple timeframes (daily, hourly, 15-minute)
- User-friendly Gradio interface
- Free stock data using yfinance API

## Hugging Face Spaces Deployment

This application is configured to run on Hugging Face Spaces. To deploy:

1. Create a new Space on Hugging Face
2. Choose "Docker" as the SDK
3. Upload all the files to your Space

## Local Development

To run locally:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Model Details

The application uses Amazon's Chronos model for time series forecasting. The model is configured to:

- Make predictions for stock prices
- Calculate confidence intervals
- Generate interactive visualizations
- Support multiple prediction horizons

## Usage

1. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
2. Select the desired timeframe (1d, 1h, 15m)
3. Choose the number of days to predict (1-30)
4. Click "Make Prediction" to generate forecasts

The application will display:
- A plot showing historical prices and predictions
- Confidence intervals for the predictions
- A separate plot showing prediction uncertainty

## License

This project is licensed under the MIT License - see the LICENSE file for details.
