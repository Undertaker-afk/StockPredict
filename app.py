import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import torch
from chronos import ChronosPipeline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import json
import spaces
import gc
import pytz
import time
import random

# Initialize global variables
pipeline = None
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform([[-1, 1]])

def retry_yfinance_request(func, max_retries=3, initial_delay=1):
    """
    Retry mechanism for yfinance requests with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
    
    Returns:
        Result of the function call if successful
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "401" in str(e) and attempt < max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise e

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_pipeline():
    """Load the Chronos model without GPU configuration"""
    global pipeline
    try:
        if pipeline is None:
            clear_gpu_memory()
            print("Loading Chronos model...")
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                device_map="auto",  # Let the model decide the best device mapping
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_safetensors=True
            )
            # Set model to evaluation mode
            pipeline.model = pipeline.model.eval()
            # Disable gradient computation
            for param in pipeline.model.parameters():
                param.requires_grad = False
            print("Chronos model loaded successfully")
        return pipeline
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def is_market_open() -> bool:
    """Check if the market is currently open"""
    now = datetime.now()
    # Check if it's a weekday (0 = Monday, 6 = Sunday)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's during market hours (9:30 AM - 4:00 PM ET)
    et_time = now.astimezone(pytz.timezone('US/Eastern'))
    market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= et_time <= market_close

def get_next_trading_day() -> datetime:
    """Get the next trading day"""
    now = datetime.now()
    next_day = now + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # Saturday or Sunday
        next_day += timedelta(days=1)
    
    return next_day

def get_historical_data(symbol: str, timeframe: str = "1d", lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch historical data using yfinance.
    
    Args:
        symbol (str): The stock symbol (e.g., 'AAPL')
        timeframe (str): The timeframe for data ('1d', '1h', '15m')
        lookback_days (int): Number of days to look back
    
    Returns:
        pd.DataFrame: Historical data with OHLCV and technical indicators
    """
    try:
        # Check if market is open for intraday data
        if timeframe in ["1h", "15m"] and not is_market_open():
            next_trading_day = get_next_trading_day()
            raise Exception(f"Market is currently closed. Next trading day is {next_trading_day.strftime('%Y-%m-%d')}")
        
        # Map timeframe to yfinance interval and adjust lookback period
        tf_map = {
            "1d": "1d",
            "1h": "1h",
            "15m": "15m"
        }
        interval = tf_map.get(timeframe, "1d")
        
        # Adjust lookback period based on timeframe
        if timeframe == "1h":
            lookback_days = min(lookback_days, 30)  # Yahoo limits hourly data to 30 days
        elif timeframe == "15m":
            lookback_days = min(lookback_days, 5)   # Yahoo limits 15m data to 5 days
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Fetch data using yfinance with retry mechanism
        ticker = yf.Ticker(symbol)
        
        def fetch_history():
            return ticker.history(start=start_date, end=end_date, interval=interval)
        
        df = retry_yfinance_request(fetch_history)
        
        if df.empty:
            raise Exception(f"No data available for {symbol} in {timeframe} timeframe")
        
        # Ensure all required columns are present and numeric
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise Exception(f"Missing required column: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get additional info for structured products with retry mechanism
        def fetch_info():
            info = ticker.info
            if info is None:
                raise Exception(f"Could not fetch company info for {symbol}")
            return info
        
        try:
            info = retry_yfinance_request(fetch_info)
            df['Market_Cap'] = float(info.get('marketCap', 0))
            df['Sector'] = info.get('sector', 'Unknown')
            df['Industry'] = info.get('industry', 'Unknown')
            df['Dividend_Yield'] = float(info.get('dividendYield', 0))
        except Exception as e:
            print(f"Warning: Could not fetch company info for {symbol}: {str(e)}")
            # Set default values for missing info
            df['Market_Cap'] = 0.0
            df['Sector'] = 'Unknown'
            df['Industry'] = 'Unknown'
            df['Dividend_Yield'] = 0.0
        
        # Calculate technical indicators with adjusted windows based on timeframe
        if timeframe == "1d":
            sma_window_20 = 20
            sma_window_50 = 50
            sma_window_200 = 200
            vol_window = 20
        elif timeframe == "1h":
            sma_window_20 = 20 * 6  # 5 trading days
            sma_window_50 = 50 * 6  # ~10 trading days
            sma_window_200 = 200 * 6  # ~40 trading days
            vol_window = 20 * 6
        else:  # 15m
            sma_window_20 = 20 * 24  # 5 trading days
            sma_window_50 = 50 * 24  # ~10 trading days
            sma_window_200 = 200 * 24  # ~40 trading days
            vol_window = 20 * 24
        
        # Calculate technical indicators
        df['SMA_20'] = df['Close'].rolling(window=sma_window_20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=sma_window_50, min_periods=1).mean()
        df['SMA_200'] = df['Close'].rolling(window=sma_window_200, min_periods=1).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        
        # Calculate returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=vol_window, min_periods=1).std()
        df['Annualized_Vol'] = df['Volatility'] * np.sqrt(252)
        
        # Calculate drawdown metrics
        df['Rolling_Max'] = df['Close'].rolling(window=len(df), min_periods=1).max()
        df['Drawdown'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
        df['Max_Drawdown'] = df['Drawdown'].rolling(window=len(df), min_periods=1).min()
        
        # Calculate liquidity metrics
        df['Avg_Daily_Volume'] = df['Volume'].rolling(window=vol_window, min_periods=1).mean()
        df['Volume_Volatility'] = df['Volume'].rolling(window=vol_window, min_periods=1).std()
        
        # Fill NaN values using forward fill then backward fill
        df = df.ffill().bfill()
        
        # Ensure we have enough data points
        min_required_points = 64  # Minimum required for Chronos
        if len(df) < min_required_points:
            # Try to fetch more historical data with retry mechanism
            extended_start_date = start_date - timedelta(days=min_required_points - len(df))
            
            def fetch_extended_history():
                return ticker.history(start=extended_start_date, end=start_date, interval=interval)
            
            extended_df = retry_yfinance_request(fetch_extended_history)
            if not extended_df.empty:
                df = pd.concat([extended_df, df])
                df = df.ffill().bfill()
        
        if len(df) < 2:
            raise Exception(f"Insufficient data points for {symbol} in {timeframe} timeframe")
        
        # Final check for any remaining None values
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error fetching historical data for {symbol}: {str(e)}")

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    # Handle None values by forward filling
    prices = prices.ffill().bfill()
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and Signal line"""
    # Handle None values by forward filling
    prices = prices.ffill().bfill()
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    # Handle None values by forward filling
    prices = prices.ffill().bfill()
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band, middle_band, lower_band

@spaces.GPU(duration=180)
def make_prediction(symbol: str, timeframe: str = "1d", prediction_days: int = 5, strategy: str = "chronos") -> Tuple[Dict, go.Figure]:
    """
    Make prediction using selected strategy with ZeroGPU.
    
    Args:
        symbol (str): Stock symbol
        timeframe (str): Data timeframe ('1d', '1h', '15m')
        prediction_days (int): Number of days to predict
        strategy (str): Prediction strategy to use
    
    Returns:
        Tuple[Dict, go.Figure]: Trading signals and visualization plot
    """
    try:
        # Get historical data
        df = get_historical_data(symbol, timeframe)
        
        if strategy == "chronos":
            try:
                # Prepare data for Chronos
                prices = df['Close'].values
                normalized_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
                
                # Ensure we have enough data points
                min_data_points = 64
                if len(normalized_prices) < min_data_points:
                    padding = np.full(min_data_points - len(normalized_prices), normalized_prices[-1])
                    normalized_prices = np.concatenate([padding, normalized_prices])
                elif len(normalized_prices) > min_data_points:
                    normalized_prices = normalized_prices[-min_data_points:]
                
                # Load pipeline and move to GPU
                pipe = load_pipeline()
                
                # Get the model's device and dtype
                device = next(pipe.model.parameters()).device
                dtype = next(pipe.model.parameters()).dtype
                print(f"Model device: {device}")
                print(f"Model dtype: {dtype}")
                
                # Convert to tensor and ensure proper shape and device
                context = torch.tensor(normalized_prices, dtype=dtype, device=device)
                
                # Adjust prediction length based on timeframe
                if timeframe == "1d":
                    max_prediction_length = 64
                elif timeframe == "1h":
                    max_prediction_length = 168
                else:  # 15m
                    max_prediction_length = 192
                
                actual_prediction_length = min(prediction_days, max_prediction_length) if timeframe == "1d" else \
                    min(prediction_days * 24, max_prediction_length) if timeframe == "1h" else \
                    min(prediction_days * 96, max_prediction_length)
                
                actual_prediction_length = max(1, actual_prediction_length)
                
                with torch.inference_mode():
                    try:
                        print(f"Attempting prediction with context shape: {context.shape}")
                        print(f"Prediction length: {actual_prediction_length}")
                        
                        # Ensure context is properly formatted for Chronos
                        if len(context.shape) == 1:
                            context = context.unsqueeze(0)
                        
                        # Verify device and dtype
                        print(f"Context device: {context.device}")
                        print(f"Context dtype: {context.dtype}")
                        print(f"Model device: {next(pipe.model.parameters()).device}")
                        print(f"Model dtype: {next(pipe.model.parameters()).dtype}")
                        
                        # Move model to evaluation mode
                        pipe.model.eval()
                        
                        # Move the entire model and all its components to GPU
                        pipe.model = pipe.model.to(device)
                        
                        # Ensure all model parameters and buffers are on GPU
                        for param in pipe.model.parameters():
                            param.data = param.data.to(device)
                        for buffer in pipe.model.buffers():
                            buffer.data = buffer.data.to(device)
                        
                        # Move any registered buffers or parameters in submodules
                        for module in pipe.model.modules():
                            if hasattr(module, 'register_buffer'):
                                for name, buffer in module._buffers.items():
                                    if buffer is not None:
                                        module._buffers[name] = buffer.to(device)
                            if hasattr(module, 'register_parameter'):
                                for name, param in module._parameters.items():
                                    if param is not None:
                                        module._parameters[name] = param.to(device)
                        
                        # Use predict_quantiles with proper formatting
                        with torch.amp.autocast('cuda'):
                            # Ensure all inputs are on GPU
                            context = context.to(device)
                            quantiles, mean = pipe.predict_quantiles(
                                context=context,
                                prediction_length=actual_prediction_length,
                                quantile_levels=[0.1, 0.5, 0.9]
                            )
                        
                        if quantiles is None or mean is None:
                            raise ValueError("Chronos returned empty prediction")
                        
                        print(f"Quantiles shape: {quantiles.shape}, Mean shape: {mean.shape}")
                        
                        # Convert to numpy arrays
                        quantiles = quantiles.detach().cpu().numpy()
                        mean = mean.detach().cpu().numpy()
                        
                        # Denormalize predictions
                        mean_pred = scaler.inverse_transform(mean.reshape(-1, 1)).flatten()
                        lower_bound = scaler.inverse_transform(quantiles[0, :, 0].reshape(-1, 1)).flatten()
                        upper_bound = scaler.inverse_transform(quantiles[0, :, 2].reshape(-1, 1)).flatten()
                        
                        # Calculate standard deviation from quantiles
                        std_pred = (upper_bound - lower_bound) / (2 * 1.645)
                        
                        # If we had to limit the prediction length, extend the prediction
                        if actual_prediction_length < prediction_days:
                            last_pred = mean_pred[-1]
                            last_std = std_pred[-1]
                            extension = np.array([last_pred * (1 + np.random.normal(0, last_std, prediction_days - actual_prediction_length))])
                            mean_pred = np.concatenate([mean_pred, extension])
                            std_pred = np.concatenate([std_pred, np.full(prediction_days - actual_prediction_length, last_std)])
                        
                    except Exception as e:
                        print(f"Chronos prediction error: {str(e)}")
                        print(f"Error type: {type(e)}")
                        print(f"Error details: {str(e)}")
                        raise
                
            except Exception as e:
                print(f"Chronos prediction failed: {str(e)}")
                print("Falling back to technical analysis")
                strategy = "technical"
        
        if strategy == "technical":
            # Technical analysis based prediction
            last_price = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            
            # Simple prediction based on technical indicators
            trend = 1 if (rsi > 50 and macd > macd_signal) else -1
            volatility = df['Volatility'].iloc[-1]
            
            # Generate predictions
            mean_pred = np.array([last_price * (1 + trend * volatility * i) for i in range(1, prediction_days + 1)])
            std_pred = np.array([volatility * last_price * i for i in range(1, prediction_days + 1)])
        
        # Create prediction dates based on timeframe
        last_date = df.index[-1]
        if timeframe == "1d":
            pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
        elif timeframe == "1h":
            pred_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=prediction_days * 24)
        else:  # 15m
            pred_dates = pd.date_range(start=last_date + timedelta(minutes=15), periods=prediction_days * 96)
        
        # Create visualization
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price Prediction', 'Technical Indicators', 'Volume'))
        
        # Add historical price
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Historical Price',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add prediction mean
        fig.add_trace(
            go.Scatter(x=pred_dates, y=mean_pred, name='Predicted Price',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(x=pred_dates, y=mean_pred + 1.96 * std_pred,
                      fill=None, mode='lines', line_color='rgba(255,0,0,0.2)',
                      name='Upper Bound'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=pred_dates, y=mean_pred - 1.96 * std_pred,
                      fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)',
                      name='Lower Bound'),
            row=1, col=1
        )
        
        # Add technical indicators
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                      line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                      line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Signal',
                      line=dict(color='green')),
            row=2, col=1
        )
        
        # Add volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                  marker_color='gray'),
            row=3, col=1
        )
        
        # Update layout with timeframe-specific settings
        fig.update_layout(
            title=f'{symbol} {timeframe} Analysis and Prediction',
            xaxis_title='Date',
            yaxis_title='Price',
            height=1000,
            showlegend=True
        )
        
        # Calculate trading signals
        signals = calculate_trading_signals(df)
        
        # Add prediction information to signals
        signals.update({
            "symbol": symbol,
            "timeframe": timeframe,
            "prediction": mean_pred.tolist(),
            "confidence": std_pred.tolist(),
            "dates": pred_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "strategy_used": strategy
        })
        
        return signals, fig
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")
    finally:
        clear_gpu_memory()

def calculate_trading_signals(df: pd.DataFrame) -> Dict:
    """Calculate trading signals based on technical indicators"""
    signals = {
        "RSI": "Oversold" if df['RSI'].iloc[-1] < 30 else "Overbought" if df['RSI'].iloc[-1] > 70 else "Neutral",
        "MACD": "Buy" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "Sell",
        "Bollinger": "Buy" if df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1] else "Sell" if df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1] else "Hold",
        "SMA": "Buy" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "Sell"
    }
    
    # Calculate overall signal
    buy_signals = sum(1 for signal in signals.values() if signal == "Buy")
    sell_signals = sum(1 for signal in signals.values() if signal == "Sell")
    
    if buy_signals > sell_signals:
        signals["Overall"] = "Buy"
    elif sell_signals > buy_signals:
        signals["Overall"] = "Sell"
    else:
        signals["Overall"] = "Hold"
    
    return signals

def create_interface():
    """Create the Gradio interface with separate tabs for different timeframes"""
    with gr.Blocks(title="Structured Product Analysis") as demo:
        gr.Markdown("# Structured Product Analysis")
        gr.Markdown("Analyze stocks for inclusion in structured financial products with extended time horizons.")
        
        # Add market status message
        market_status = "Market is currently closed" if not is_market_open() else "Market is currently open"
        next_trading_day = get_next_trading_day()
        gr.Markdown(f"""
        ### Market Status: {market_status}
        Next trading day: {next_trading_day.strftime('%Y-%m-%d')}
        """)
        
        with gr.Tabs() as tabs:
            # Daily Analysis Tab
            with gr.TabItem("Daily Analysis"):
                with gr.Row():
                    with gr.Column():
                        daily_symbol = gr.Textbox(label="Stock Symbol (e.g., AAPL)", value="AAPL")
                        daily_prediction_days = gr.Slider(
                            minimum=1,
                            maximum=365,
                            value=30,
                            step=1,
                            label="Days to Predict"
                        )
                        daily_lookback_days = gr.Slider(
                            minimum=1,
                            maximum=3650,
                            value=365,
                            step=1,
                            label="Historical Lookback (Days)"
                        )
                        daily_strategy = gr.Dropdown(
                            choices=["chronos", "technical"],
                            label="Prediction Strategy",
                            value="chronos"
                        )
                        daily_predict_btn = gr.Button("Analyze Stock")
                    
                    with gr.Column():
                        daily_plot = gr.Plot(label="Analysis and Prediction")
                
                with gr.Row():
                    with gr.Column():

                        gr.Markdown("### Structured Product Metrics")
                        daily_metrics = gr.JSON(label="Product Metrics")
                        
                        gr.Markdown("### Risk Analysis")
                        daily_risk_metrics = gr.JSON(label="Risk Metrics")
                        
                        gr.Markdown("### Sector Analysis")
                        daily_sector_metrics = gr.JSON(label="Sector Metrics")

                        gr.Markdown("### Trading Signals")
                        daily_signals = gr.JSON(label="Trading Signals")
            
            # Hourly Analysis Tab
            with gr.TabItem("Hourly Analysis"):
                with gr.Row():
                    with gr.Column():
                        hourly_symbol = gr.Textbox(label="Stock Symbol (e.g., AAPL)", value="AAPL")
                        hourly_prediction_days = gr.Slider(
                            minimum=1,
                            maximum=7,  # Limited to 7 days for hourly predictions
                            value=3,
                            step=1,
                            label="Days to Predict"
                        )
                        hourly_lookback_days = gr.Slider(
                            minimum=1,
                            maximum=30,  # Limited to 30 days for hourly data
                            value=14,
                            step=1,
                            label="Historical Lookback (Days)"
                        )
                        hourly_strategy = gr.Dropdown(
                            choices=["chronos", "technical"],
                            label="Prediction Strategy",
                            value="chronos"
                        )
                        hourly_predict_btn = gr.Button("Analyze Stock")
                        gr.Markdown("""
                        **Note for Hourly Analysis:**
                        - Maximum lookback period: 30 days (Yahoo Finance limit)
                        - Maximum prediction period: 7 days
                        - Data is only available during market hours
                        """)
                    
                    with gr.Column():
                        hourly_plot = gr.Plot(label="Analysis and Prediction")
                        hourly_signals = gr.JSON(label="Trading Signals")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Structured Product Metrics")
                        hourly_metrics = gr.JSON(label="Product Metrics")
                        
                        gr.Markdown("### Risk Analysis")
                        hourly_risk_metrics = gr.JSON(label="Risk Metrics")
                        
                        gr.Markdown("### Sector Analysis")
                        hourly_sector_metrics = gr.JSON(label="Sector Metrics")
            
            # 15-Minute Analysis Tab
            with gr.TabItem("15-Minute Analysis"):
                with gr.Row():
                    with gr.Column():
                        min15_symbol = gr.Textbox(label="Stock Symbol (e.g., AAPL)", value="AAPL")
                        min15_prediction_days = gr.Slider(
                            minimum=1,
                            maximum=2,  # Limited to 2 days for 15-minute predictions
                            value=1,
                            step=1,
                            label="Days to Predict"
                        )
                        min15_lookback_days = gr.Slider(
                            minimum=1,
                            maximum=5,  # Yahoo Finance limit for 15-minute data
                            value=3,
                            step=1,
                            label="Historical Lookback (Days)"
                        )
                        min15_strategy = gr.Dropdown(
                            choices=["chronos", "technical"],
                            label="Prediction Strategy",
                            value="chronos"
                        )
                        min15_predict_btn = gr.Button("Analyze Stock")
                        gr.Markdown("""
                        **Note for 15-Minute Analysis:**
                        - Maximum lookback period: 5 days (Yahoo Finance limit)
                        - Maximum prediction period: 2 days
                        - Data is only available during market hours
                        - Requires at least 64 data points for Chronos predictions
                        """)
                    
                    with gr.Column():
                        min15_plot = gr.Plot(label="Analysis and Prediction")
                        min15_signals = gr.JSON(label="Trading Signals")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Structured Product Metrics")
                        min15_metrics = gr.JSON(label="Product Metrics")
                        
                        gr.Markdown("### Risk Analysis")
                        min15_risk_metrics = gr.JSON(label="Risk Metrics")
                        
                        gr.Markdown("### Sector Analysis")
                        min15_sector_metrics = gr.JSON(label="Sector Metrics")
        
        def analyze_stock(symbol, timeframe, prediction_days, lookback_days, strategy):
            try:
                signals, fig = make_prediction(symbol, timeframe, prediction_days, strategy)
                
                # Get historical data for additional metrics
                df = get_historical_data(symbol, timeframe, lookback_days)
                
                # Calculate structured product metrics
                product_metrics = {
                    "Market_Cap": df['Market_Cap'].iloc[-1],
                    "Sector": df['Sector'].iloc[-1],
                    "Industry": df['Industry'].iloc[-1],
                    "Dividend_Yield": df['Dividend_Yield'].iloc[-1],
                    "Avg_Daily_Volume": df['Avg_Daily_Volume'].iloc[-1],
                    "Volume_Volatility": df['Volume_Volatility'].iloc[-1]
                }
                
                # Calculate risk metrics
                risk_metrics = {
                    "Annualized_Volatility": df['Annualized_Vol'].iloc[-1],
                    "Max_Drawdown": df['Max_Drawdown'].iloc[-1],
                    "Current_Drawdown": df['Drawdown'].iloc[-1],
                    "Sharpe_Ratio": (df['Returns'].mean() * 252) / (df['Returns'].std() * np.sqrt(252)),
                    "Sortino_Ratio": (df['Returns'].mean() * 252) / (df['Returns'][df['Returns'] < 0].std() * np.sqrt(252))
                }
                
                # Calculate sector metrics
                sector_metrics = {
                    "Sector": df['Sector'].iloc[-1],
                    "Industry": df['Industry'].iloc[-1],
                    "Market_Cap_Rank": "Large" if df['Market_Cap'].iloc[-1] > 1e10 else "Mid" if df['Market_Cap'].iloc[-1] > 1e9 else "Small",
                    "Liquidity_Score": "High" if df['Avg_Daily_Volume'].iloc[-1] > 1e6 else "Medium" if df['Avg_Daily_Volume'].iloc[-1] > 1e5 else "Low"
                }
                
                return signals, fig, product_metrics, risk_metrics, sector_metrics
            except Exception as e:
                error_message = str(e)
                if "Market is currently closed" in error_message:
                    error_message = f"{error_message}. Please try again during market hours or use daily timeframe."
                elif "Insufficient data points" in error_message:
                    error_message = f"Not enough data available for {symbol} in {timeframe} timeframe. Please try a different timeframe or symbol."
                elif "no price data found" in error_message:
                    error_message = f"No data available for {symbol} in {timeframe} timeframe. Please try a different timeframe or symbol."
                raise gr.Error(error_message)
        
        # Daily analysis button click
        def daily_analysis(s: str, pd: int, ld: int, st: str) -> Tuple[Dict, go.Figure, Dict, Dict, Dict]:
            """
            Process daily timeframe stock analysis and generate predictions.

            Args:
                s (str): Stock symbol (e.g., "AAPL", "MSFT", "GOOGL")
                pd (int): Number of days to predict (1-365)
                ld (int): Historical lookback period in days (1-3650)
                st (str): Prediction strategy to use ("chronos" or "technical")

            Returns:
                Tuple[Dict, go.Figure, Dict, Dict, Dict]: A tuple containing:
                    - Trading signals dictionary
                    - Plotly figure with price and technical analysis
                    - Product metrics dictionary
                    - Risk metrics dictionary
                    - Sector metrics dictionary

            Example:
                >>> daily_analysis("AAPL", 30, 365, "chronos")
                ({'RSI': 'Neutral', 'MACD': 'Buy', ...}, <Figure>, {...}, {...}, {...})
            """
            return analyze_stock(s, "1d", pd, ld, st)

        daily_predict_btn.click(
            fn=daily_analysis,
            inputs=[daily_symbol, daily_prediction_days, daily_lookback_days, daily_strategy],
            outputs=[daily_signals, daily_plot, daily_metrics, daily_risk_metrics, daily_sector_metrics]
        )
        
        # Hourly analysis button click
        def hourly_analysis(s: str, pd: int, ld: int, st: str) -> Tuple[Dict, go.Figure, Dict, Dict, Dict]:
            """
            Process hourly timeframe stock analysis and generate predictions.

            Args:
                s (str): Stock symbol (e.g., "AAPL", "MSFT", "GOOGL")
                pd (int): Number of days to predict (1-7)
                ld (int): Historical lookback period in days (1-30)
                st (str): Prediction strategy to use ("chronos" or "technical")

            Returns:
                Tuple[Dict, go.Figure, Dict, Dict, Dict]: A tuple containing:
                    - Trading signals dictionary
                    - Plotly figure with price and technical analysis
                    - Product metrics dictionary
                    - Risk metrics dictionary
                    - Sector metrics dictionary

            Example:
                >>> hourly_analysis("AAPL", 3, 14, "chronos")
                ({'RSI': 'Neutral', 'MACD': 'Buy', ...}, <Figure>, {...}, {...}, {...})
            """
            return analyze_stock(s, "1h", pd, ld, st)

        hourly_predict_btn.click(
            fn=hourly_analysis,
            inputs=[hourly_symbol, hourly_prediction_days, hourly_lookback_days, hourly_strategy],
            outputs=[hourly_signals, hourly_plot, hourly_metrics, hourly_risk_metrics, hourly_sector_metrics]
        )
        
        # 15-minute analysis button click
        def min15_analysis(s: str, pd: int, ld: int, st: str) -> Tuple[Dict, go.Figure, Dict, Dict, Dict]:
            """
            Process 15-minute timeframe stock analysis and generate predictions.

            Args:
                s (str): Stock symbol (e.g., "AAPL", "MSFT", "GOOGL")
                pd (int): Number of days to predict (1-2)
                ld (int): Historical lookback period in days (1-5)
                st (str): Prediction strategy to use ("chronos" or "technical")

            Returns:
                Tuple[Dict, go.Figure, Dict, Dict, Dict]: A tuple containing:
                    - Trading signals dictionary
                    - Plotly figure with price and technical analysis
                    - Product metrics dictionary
                    - Risk metrics dictionary
                    - Sector metrics dictionary

            Example:
                >>> min15_analysis("AAPL", 1, 3, "chronos")
                ({'RSI': 'Neutral', 'MACD': 'Buy', ...}, <Figure>, {...}, {...}, {...})
            """
            return analyze_stock(s, "15m", pd, ld, st)

        min15_predict_btn.click(
            fn=min15_analysis,
            inputs=[min15_symbol, min15_prediction_days, min15_lookback_days, min15_strategy],
            outputs=[min15_signals, min15_plot, min15_metrics, min15_risk_metrics, min15_sector_metrics]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(ssr_mode=False, mcp_server=True) 