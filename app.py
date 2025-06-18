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

@spaces.GPU()
def load_pipeline():
    """Load the Chronos model without GPU configuration"""
    global pipeline
    try:
        if pipeline is None:
            clear_gpu_memory()
            print("Loading Chronos model...")
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                device_map="cuda",  # Force CUDA device mapping
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
    Fetch historical data using yfinance with enhanced support for intraday data.
    
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
        
        # Adjust lookback period based on timeframe and yfinance limits
        if timeframe == "1h":
            lookback_days = min(lookback_days, 60)  # Yahoo allows up to 60 days for hourly data
        elif timeframe == "15m":
            lookback_days = min(lookback_days, 7)   # Yahoo allows up to 7 days for 15m data
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Fetch data using yfinance with retry mechanism
        ticker = yf.Ticker(symbol)
        
        def fetch_history():
            return ticker.history(
                start=start_date, 
                end=end_date, 
                interval=interval,
                prepost=True,  # Include pre/post market data for intraday
                actions=True,  # Include dividends and splits
                auto_adjust=True,  # Automatically adjust for splits
                back_adjust=True,  # Back-adjust data for splits
                repair=True  # Repair missing data points
            )
        
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
            
            # Add additional company metrics
            df['Enterprise_Value'] = float(info.get('enterpriseValue', 0))
            df['P/E_Ratio'] = float(info.get('trailingPE', 0))
            df['Forward_P/E'] = float(info.get('forwardPE', 0))
            df['PEG_Ratio'] = float(info.get('pegRatio', 0))
            df['Price_to_Book'] = float(info.get('priceToBook', 0))
            df['Price_to_Sales'] = float(info.get('priceToSalesTrailing12Months', 0))
            df['Return_on_Equity'] = float(info.get('returnOnEquity', 0))
            df['Return_on_Assets'] = float(info.get('returnOnAssets', 0))
            df['Debt_to_Equity'] = float(info.get('debtToEquity', 0))
            df['Current_Ratio'] = float(info.get('currentRatio', 0))
            df['Quick_Ratio'] = float(info.get('quickRatio', 0))
            df['Gross_Margin'] = float(info.get('grossMargins', 0))
            df['Operating_Margin'] = float(info.get('operatingMargins', 0))
            df['Net_Margin'] = float(info.get('netIncomeToCommon', 0))
            
        except Exception as e:
            print(f"Warning: Could not fetch company info for {symbol}: {str(e)}")
            # Set default values for missing info
            df['Market_Cap'] = 0.0
            df['Sector'] = 'Unknown'
            df['Industry'] = 'Unknown'
            df['Dividend_Yield'] = 0.0
            df['Enterprise_Value'] = 0.0
            df['P/E_Ratio'] = 0.0
            df['Forward_P/E'] = 0.0
            df['PEG_Ratio'] = 0.0
            df['Price_to_Book'] = 0.0
            df['Price_to_Sales'] = 0.0
            df['Return_on_Equity'] = 0.0
            df['Return_on_Assets'] = 0.0
            df['Debt_to_Equity'] = 0.0
            df['Current_Ratio'] = 0.0
            df['Quick_Ratio'] = 0.0
            df['Gross_Margin'] = 0.0
            df['Operating_Margin'] = 0.0
            df['Net_Margin'] = 0.0
        
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
        
        # Calculate additional intraday metrics for shorter timeframes
        if timeframe in ["1h", "15m"]:
            # Intraday volatility
            df['Intraday_High_Low'] = (df['High'] - df['Low']) / df['Close']
            df['Intraday_Volatility'] = df['Intraday_High_Low'].rolling(window=vol_window, min_periods=1).mean()
            
            # Volume analysis
            df['Volume_Price_Trend'] = (df['Volume'] * df['Returns']).rolling(window=vol_window, min_periods=1).sum()
            df['Volume_SMA'] = df['Volume'].rolling(window=vol_window, min_periods=1).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Price momentum
            df['Price_Momentum'] = df['Close'].pct_change(periods=5)
            df['Volume_Momentum'] = df['Volume'].pct_change(periods=5)
        
        # Fill NaN values using forward fill then backward fill
        df = df.ffill().bfill()
        
        # Ensure we have enough data points
        min_required_points = 64  # Minimum required for Chronos
        if len(df) < min_required_points:
            # Try to fetch more historical data with retry mechanism
            extended_start_date = start_date - timedelta(days=min_required_points - len(df))
            
            def fetch_extended_history():
                return ticker.history(
                    start=extended_start_date, 
                    end=start_date, 
                    interval=interval,
                    prepost=True,
                    actions=True,
                    auto_adjust=True,
                    back_adjust=True,
                    repair=True
                )
            
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
                device = torch.device("cuda:0")  # Force CUDA device
                dtype = torch.float16  # Force float16
                print(f"Model device: {device}")
                print(f"Model dtype: {dtype}")
                
                # Convert to tensor and ensure proper shape and device
                context = torch.tensor(normalized_prices, dtype=dtype, device=device)
                
                # Adjust prediction length based on timeframe
                if timeframe == "1d":
                    max_prediction_length = 64  # Chronos maximum
                    window_size = 64  # Use full context window
                elif timeframe == "1h":
                    max_prediction_length = 64  # Chronos maximum
                    window_size = 64  # Use full context window
                else:  # 15m
                    max_prediction_length = 64  # Chronos maximum
                    window_size = 64  # Use full context window
                
                # Calculate actual prediction length based on timeframe
                if timeframe == "1d":
                    actual_prediction_length = min(prediction_days, max_prediction_length)
                elif timeframe == "1h":
                    actual_prediction_length = min(prediction_days * 24, max_prediction_length)
                else:  # 15m
                    actual_prediction_length = min(prediction_days * 96, max_prediction_length)
                
                actual_prediction_length = max(1, actual_prediction_length)
                
                # Use predict_quantiles with proper formatting
                with torch.amp.autocast('cuda'):
                    # Ensure all inputs are on GPU
                    context = context.to(device)
                    
                    # Move quantile levels to GPU
                    quantile_levels = torch.tensor([0.1, 0.5, 0.9], device=device, dtype=dtype)
                    
                    # Ensure prediction length is on GPU
                    prediction_length = torch.tensor(actual_prediction_length, device=device, dtype=torch.long)
                    
                    # Force all model components to GPU
                    pipe.model = pipe.model.to(device)
                    
                    # Move model to evaluation mode
                    pipe.model.eval()
                    
                    # Ensure context is properly shaped and on GPU
                    if len(context.shape) == 1:
                        context = context.unsqueeze(0)
                    context = context.to(device)
                    
                    # Move all model parameters and buffers to GPU
                    for param in pipe.model.parameters():
                        param.data = param.data.to(device)
                    for buffer in pipe.model.buffers():
                        buffer.data = buffer.data.to(device)
                    
                    # Move all model submodules to GPU
                    for module in pipe.model.modules():
                        if hasattr(module, 'to'):
                            module.to(device)
                    
                    # Move all model attributes to GPU
                    for name, value in pipe.model.__dict__.items():
                        if isinstance(value, torch.Tensor):
                            pipe.model.__dict__[name] = value.to(device)
                    
                    # Move all model config tensors to GPU
                    if hasattr(pipe.model, 'config'):
                        for key, value in pipe.model.config.__dict__.items():
                            if isinstance(value, torch.Tensor):
                                setattr(pipe.model.config, key, value.to(device))
                    
                    # Move all pipeline tensors to GPU
                    for name, value in pipe.__dict__.items():
                        if isinstance(value, torch.Tensor):
                            setattr(pipe, name, value.to(device))
                    
                    # Ensure all model states are on GPU
                    if hasattr(pipe.model, 'state_dict'):
                        state_dict = pipe.model.state_dict()
                        for key in state_dict:
                            if isinstance(state_dict[key], torch.Tensor):
                                state_dict[key] = state_dict[key].to(device)
                        pipe.model.load_state_dict(state_dict)
                    
                    # Move any additional components to GPU
                    if hasattr(pipe, 'tokenizer'):
                        # Move tokenizer to GPU if it supports it
                        if hasattr(pipe.tokenizer, 'to'):
                            pipe.tokenizer = pipe.tokenizer.to(device)
                        
                        # Move all tokenizer tensors to GPU
                        for name, value in pipe.tokenizer.__dict__.items():
                            if isinstance(value, torch.Tensor):
                                setattr(pipe.tokenizer, name, value.to(device))
                        
                        # Handle MeanScaleUniformBins specific attributes
                        if hasattr(pipe.tokenizer, 'bins'):
                            if isinstance(pipe.tokenizer.bins, torch.Tensor):
                                pipe.tokenizer.bins = pipe.tokenizer.bins.to(device)
                        
                        if hasattr(pipe.tokenizer, 'scale'):
                            if isinstance(pipe.tokenizer.scale, torch.Tensor):
                                pipe.tokenizer.scale = pipe.tokenizer.scale.to(device)
                        
                        if hasattr(pipe.tokenizer, 'mean'):
                            if isinstance(pipe.tokenizer.mean, torch.Tensor):
                                pipe.tokenizer.mean = pipe.tokenizer.mean.to(device)
                        
                        # Move any additional tensors in the tokenizer's attributes to GPU
                        for name, value in pipe.tokenizer.__dict__.items():
                            if isinstance(value, torch.Tensor):
                                pipe.tokenizer.__dict__[name] = value.to(device)
                        
                        # Remove the EOS token handling since MeanScaleUniformBins doesn't use it
                        if hasattr(pipe.tokenizer, '_append_eos_token'):
                            # Create a wrapper that just returns the input tensors
                            def wrapped_append_eos(token_ids, attention_mask):
                                return token_ids, attention_mask
                            pipe.tokenizer._append_eos_token = wrapped_append_eos
                    
                    # Force synchronization again to ensure all tensors are on GPU
                    torch.cuda.synchronize()
                    
                    # Ensure all model components are in eval mode
                    pipe.model.eval()
                    
                    # Move any additional tensors in the model's config to GPU
                    if hasattr(pipe.model, 'config'):
                        for key, value in pipe.model.config.__dict__.items():
                            if isinstance(value, torch.Tensor):
                                setattr(pipe.model.config, key, value.to(device))
                    
                    # Move any additional tensors in the model's state dict to GPU
                    if hasattr(pipe.model, 'state_dict'):
                        state_dict = pipe.model.state_dict()
                        for key in state_dict:
                            if isinstance(state_dict[key], torch.Tensor):
                                state_dict[key] = state_dict[key].to(device)
                        pipe.model.load_state_dict(state_dict)
                    
                    # Move any additional tensors in the model's buffers to GPU
                    for name, buffer in pipe.model.named_buffers():
                        if buffer is not None:
                            pipe.model.register_buffer(name, buffer.to(device))
                    
                    # Move any additional tensors in the model's parameters to GPU
                    for name, param in pipe.model.named_parameters():
                        if param is not None:
                            param.data = param.data.to(device)
                    
                    # Move any additional tensors in the model's attributes to GPU
                    for name, value in pipe.model.__dict__.items():
                        if isinstance(value, torch.Tensor):
                            pipe.model.__dict__[name] = value.to(device)
                    
                    # Move any additional tensors in the model's modules to GPU
                    for name, module in pipe.model.named_modules():
                        if hasattr(module, 'to'):
                            module.to(device)
                        # Move any tensors in the module's __dict__
                        for key, value in module.__dict__.items():
                            if isinstance(value, torch.Tensor):
                                setattr(module, key, value.to(device))
                    
                    # Force synchronization again to ensure all tensors are on GPU
                    torch.cuda.synchronize()
                    
                    # Ensure tokenizer is on GPU and all its tensors are on GPU
                    if hasattr(pipe, 'tokenizer'):
                        # Move tokenizer to GPU if it supports it
                        if hasattr(pipe.tokenizer, 'to'):
                            pipe.tokenizer = pipe.tokenizer.to(device)
                        
                        # Move all tokenizer tensors to GPU
                        for name, value in pipe.tokenizer.__dict__.items():
                            if isinstance(value, torch.Tensor):
                                setattr(pipe.tokenizer, name, value.to(device))
                        
                        # Handle MeanScaleUniformBins specific attributes
                        if hasattr(pipe.tokenizer, 'bins'):
                            if isinstance(pipe.tokenizer.bins, torch.Tensor):
                                pipe.tokenizer.bins = pipe.tokenizer.bins.to(device)
                        
                        if hasattr(pipe.tokenizer, 'scale'):
                            if isinstance(pipe.tokenizer.scale, torch.Tensor):
                                pipe.tokenizer.scale = pipe.tokenizer.scale.to(device)
                        
                        if hasattr(pipe.tokenizer, 'mean'):
                            if isinstance(pipe.tokenizer.mean, torch.Tensor):
                                pipe.tokenizer.mean = pipe.tokenizer.mean.to(device)
                        
                        # Move any additional tensors in the tokenizer's attributes to GPU
                        for name, value in pipe.tokenizer.__dict__.items():
                            if isinstance(value, torch.Tensor):
                                pipe.tokenizer.__dict__[name] = value.to(device)
                    
                    # Force synchronization again to ensure all tensors are on GPU
                    torch.cuda.synchronize()
                    
                    # Make prediction
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
                    # Initialize arrays for extended predictions
                    extended_mean_pred = mean_pred.copy()
                    extended_std_pred = std_pred.copy()
                    
                    # Calculate the number of extension steps needed
                    remaining_days = prediction_days - actual_prediction_length
                    steps_needed = (remaining_days + actual_prediction_length - 1) // actual_prediction_length
                    
                    for step in range(steps_needed):
                        # Use the last window_size points as context for next prediction
                        context_window = extended_mean_pred[-window_size:]
                        
                        # Normalize the context window
                        normalized_context = scaler.fit_transform(context_window.reshape(-1, 1)).flatten()
                        
                        # Convert to tensor and ensure proper shape
                        context = torch.tensor(normalized_context, dtype=dtype, device=device)
                        if len(context.shape) == 1:
                            context = context.unsqueeze(0)
                        
                        # Calculate next prediction length based on timeframe
                        if timeframe == "1d":
                            next_length = min(max_prediction_length, remaining_days)
                        elif timeframe == "1h":
                            next_length = min(max_prediction_length, remaining_days * 24)
                        else:  # 15m
                            next_length = min(max_prediction_length, remaining_days * 96)
                        
                        # Make prediction for next window
                        with torch.amp.autocast('cuda'):
                            next_quantiles, next_mean = pipe.predict_quantiles(
                                context=context,
                                prediction_length=next_length,
                                quantile_levels=[0.1, 0.5, 0.9]
                            )
                        
                        # Convert predictions to numpy and denormalize
                        next_mean = next_mean.detach().cpu().numpy()
                        next_quantiles = next_quantiles.detach().cpu().numpy()
                        
                        # Denormalize predictions
                        next_mean_pred = scaler.inverse_transform(next_mean.reshape(-1, 1)).flatten()
                        next_lower = scaler.inverse_transform(next_quantiles[0, :, 0].reshape(-1, 1)).flatten()
                        next_upper = scaler.inverse_transform(next_quantiles[0, :, 2].reshape(-1, 1)).flatten()
                        
                        # Calculate standard deviation
                        next_std_pred = (next_upper - next_lower) / (2 * 1.645)
                        
                        # Apply exponential smoothing to reduce prediction drift
                        if step > 0:
                            alpha = 0.3  # Smoothing factor
                            next_mean_pred = alpha * next_mean_pred + (1 - alpha) * extended_mean_pred[-len(next_mean_pred):]
                            next_std_pred = alpha * next_std_pred + (1 - alpha) * extended_std_pred[-len(next_std_pred):]
                        
                        # Append predictions
                        extended_mean_pred = np.concatenate([extended_mean_pred, next_mean_pred])
                        extended_std_pred = np.concatenate([extended_std_pred, next_std_pred])
                        
                        # Update remaining days
                        if timeframe == "1d":
                            remaining_days -= len(next_mean_pred)
                        elif timeframe == "1h":
                            remaining_days -= len(next_mean_pred) / 24
                        else:  # 15m
                            remaining_days -= len(next_mean_pred) / 96
                        
                        if remaining_days <= 0:
                            break
                    
                    # Trim to exact prediction length if needed
                    if timeframe == "1d":
                        mean_pred = extended_mean_pred[:prediction_days]
                        std_pred = extended_std_pred[:prediction_days]
                    elif timeframe == "1h":
                        mean_pred = extended_mean_pred[:prediction_days * 24]
                        std_pred = extended_std_pred[:prediction_days * 24]
                    else:  # 15m
                        mean_pred = extended_mean_pred[:prediction_days * 96]
                        std_pred = extended_std_pred[:prediction_days * 96]
                
                # Extend Chronos forecasting to volume and technical indicators
                volume_pred = None
                rsi_pred = None
                macd_pred = None
                
                try:
                    # Prepare volume data for Chronos
                    volume_data = df['Volume'].values
                    if len(volume_data) >= 64:
                        # Normalize volume data
                        volume_scaler = MinMaxScaler(feature_range=(-1, 1))
                        normalized_volume = volume_scaler.fit_transform(volume_data.reshape(-1, 1)).flatten()
                        
                        # Use last 64 points for volume prediction
                        volume_context = normalized_volume[-64:]
                        volume_context_tensor = torch.tensor(volume_context, dtype=dtype, device=device)
                        if len(volume_context_tensor.shape) == 1:
                            volume_context_tensor = volume_context_tensor.unsqueeze(0)
                        
                        # Predict volume
                        with torch.amp.autocast('cuda'):
                            volume_quantiles, volume_mean = pipe.predict_quantiles(
                                context=volume_context_tensor,
                                prediction_length=min(actual_prediction_length, 64),
                                quantile_levels=[0.1, 0.5, 0.9]
                            )
                        
                        # Convert and denormalize volume predictions
                        volume_mean = volume_mean.detach().cpu().numpy()
                        volume_pred = volume_scaler.inverse_transform(volume_mean.reshape(-1, 1)).flatten()
                        
                        # Extend volume predictions if needed
                        if len(volume_pred) < len(mean_pred):
                            last_volume = volume_pred[-1]
                            extension_length = len(mean_pred) - len(volume_pred)
                            volume_extension = np.full(extension_length, last_volume)
                            volume_pred = np.concatenate([volume_pred, volume_extension])
                except Exception as e:
                    print(f"Volume prediction error: {str(e)}")
                    # Fallback: use historical average
                    avg_volume = df['Volume'].mean()
                    volume_pred = np.full(len(mean_pred), avg_volume)
                
                try:
                    # Prepare RSI data for Chronos
                    rsi_data = df['RSI'].values
                    if len(rsi_data) >= 64 and not np.any(np.isnan(rsi_data)):
                        # RSI is already normalized (0-100), but we'll scale it to (-1, 1)
                        rsi_scaler = MinMaxScaler(feature_range=(-1, 1))
                        normalized_rsi = rsi_scaler.fit_transform(rsi_data.reshape(-1, 1)).flatten()
                        
                        # Use last 64 points for RSI prediction
                        rsi_context = normalized_rsi[-64:]
                        rsi_context_tensor = torch.tensor(rsi_context, dtype=dtype, device=device)
                        if len(rsi_context_tensor.shape) == 1:
                            rsi_context_tensor = rsi_context_tensor.unsqueeze(0)
                        
                        # Predict RSI
                        with torch.amp.autocast('cuda'):
                            rsi_quantiles, rsi_mean = pipe.predict_quantiles(
                                context=rsi_context_tensor,
                                prediction_length=min(actual_prediction_length, 64),
                                quantile_levels=[0.1, 0.5, 0.9]
                            )
                        
                        # Convert and denormalize RSI predictions
                        rsi_mean = rsi_mean.detach().cpu().numpy()
                        rsi_pred = rsi_scaler.inverse_transform(rsi_mean.reshape(-1, 1)).flatten()
                        
                        # Clamp RSI to valid range (0-100)
                        rsi_pred = np.clip(rsi_pred, 0, 100)
                        
                        # Extend RSI predictions if needed
                        if len(rsi_pred) < len(mean_pred):
                            last_rsi = rsi_pred[-1]
                            extension_length = len(mean_pred) - len(rsi_pred)
                            rsi_extension = np.full(extension_length, last_rsi)
                            rsi_pred = np.concatenate([rsi_pred, rsi_extension])
                except Exception as e:
                    print(f"RSI prediction error: {str(e)}")
                    # Fallback: use last known RSI value
                    last_rsi = df['RSI'].iloc[-1]
                    rsi_pred = np.full(len(mean_pred), last_rsi)
                
                try:
                    # Prepare MACD data for Chronos
                    macd_data = df['MACD'].values
                    if len(macd_data) >= 64 and not np.any(np.isnan(macd_data)):
                        # Normalize MACD data
                        macd_scaler = MinMaxScaler(feature_range=(-1, 1))
                        normalized_macd = macd_scaler.fit_transform(macd_data.reshape(-1, 1)).flatten()
                        
                        # Use last 64 points for MACD prediction
                        macd_context = normalized_macd[-64:]
                        macd_context_tensor = torch.tensor(macd_context, dtype=dtype, device=device)
                        if len(macd_context_tensor.shape) == 1:
                            macd_context_tensor = macd_context_tensor.unsqueeze(0)
                        
                        # Predict MACD
                        with torch.amp.autocast('cuda'):
                            macd_quantiles, macd_mean = pipe.predict_quantiles(
                                context=macd_context_tensor,
                                prediction_length=min(actual_prediction_length, 64),
                                quantile_levels=[0.1, 0.5, 0.9]
                            )
                        
                        # Convert and denormalize MACD predictions
                        macd_mean = macd_mean.detach().cpu().numpy()
                        macd_pred = macd_scaler.inverse_transform(macd_mean.reshape(-1, 1)).flatten()
                        
                        # Extend MACD predictions if needed
                        if len(macd_pred) < len(mean_pred):
                            last_macd = macd_pred[-1]
                            extension_length = len(mean_pred) - len(macd_pred)
                            macd_extension = np.full(extension_length, last_macd)
                            macd_pred = np.concatenate([macd_pred, macd_extension])
                except Exception as e:
                    print(f"MACD prediction error: {str(e)}")
                    # Fallback: use last known MACD value
                    last_macd = df['MACD'].iloc[-1]
                    macd_pred = np.full(len(mean_pred), last_macd)
                
            except Exception as e:
                print(f"Chronos prediction error: {str(e)}")
                print(f"Error type: {type(e)}")
                print(f"Error details: {str(e)}")
                raise
        
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
        
        # Add predicted technical indicators if available
        if rsi_pred is not None:
            fig.add_trace(
                go.Scatter(x=pred_dates, y=rsi_pred, name='Predicted RSI',
                          line=dict(color='purple', dash='dash')),
                row=2, col=1
            )
        
        if macd_pred is not None:
            fig.add_trace(
                go.Scatter(x=pred_dates, y=macd_pred, name='Predicted MACD',
                          line=dict(color='orange', dash='dash')),
                row=2, col=1
            )
        
        # Add volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                  marker_color='gray'),
            row=3, col=1
        )
        
        # Add predicted volume if available
        if volume_pred is not None:
            fig.add_trace(
                go.Bar(x=pred_dates, y=volume_pred, name='Predicted Volume',
                      marker_color='red', opacity=0.7),
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
        
        # Add predicted indicators to signals if available
        if volume_pred is not None:
            signals["predicted_volume"] = volume_pred.tolist()
        if rsi_pred is not None:
            signals["predicted_rsi"] = rsi_pred.tolist()
        if macd_pred is not None:
            signals["predicted_macd"] = macd_pred.tolist()
        
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
                            maximum=60,  # Enhanced to 60 days for hourly data
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
                        **Hourly Analysis Features:**
                        - **Extended Data Range**: Up to 60 days of historical data
                        - **Pre/Post Market Data**: Includes extended hours trading data
                        - **Auto-Adjusted Data**: Automatically adjusted for splits and dividends
                        - **Metrics**: Intraday volatility, volume analysis, and momentum indicators
                        - **Comprehensive Financial Ratios**: P/E, PEG, Price-to-Book, and more
                        - **Maximum prediction period**: 7 days
                        - **Data available during market hours only**
                        """)
                    
                    with gr.Column():
                        hourly_plot = gr.Plot(label="Analysis and Prediction")
                        hourly_signals = gr.JSON(label="Trading Signals")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Structured Product Metrics")
                        hourly_metrics = gr.JSON(label="Product Metrics")
                        
                        gr.Markdown("### Comprehensive Risk Analysis")
                        hourly_risk_metrics = gr.JSON(label="Risk Metrics")
                        
                        gr.Markdown("### Sector & Financial Analysis")
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
                            maximum=7,  # 7 days for 15-minute data
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
                        **15-Minute Analysis Features:**
                        - **Data Range**: Up to 7 days of historical data (vs 5 days previously)
                        - **High-Frequency Metrics**: Intraday volatility, volume-price trends, momentum analysis
                        - **Pre/Post Market Data**: Includes extended hours trading data
                        - **Auto-Adjusted Data**: Automatically adjusted for splits and dividends
                        - **Enhanced Technical Indicators**: Optimized for short-term trading
                        - **Maximum prediction period**: 2 days
                        - **Requires at least 64 data points for Chronos predictions**
                        - **Data available during market hours only**
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
                        
                        gr.Markdown("### Sector & Financial Analysis")
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
                    "Volume_Volatility": df['Volume_Volatility'].iloc[-1],
                    "Enterprise_Value": df['Enterprise_Value'].iloc[-1],
                    "P/E_Ratio": df['P/E_Ratio'].iloc[-1],
                    "Forward_P/E": df['Forward_P/E'].iloc[-1],
                    "PEG_Ratio": df['PEG_Ratio'].iloc[-1],
                    "Price_to_Book": df['Price_to_Book'].iloc[-1],
                    "Price_to_Sales": df['Price_to_Sales'].iloc[-1]
                }
                
                # Calculate risk metrics
                risk_metrics = {
                    "Annualized_Volatility": df['Annualized_Vol'].iloc[-1],
                    "Max_Drawdown": df['Max_Drawdown'].iloc[-1],
                    "Current_Drawdown": df['Drawdown'].iloc[-1],
                    "Sharpe_Ratio": (df['Returns'].mean() * 252) / (df['Returns'].std() * np.sqrt(252)),
                    "Sortino_Ratio": (df['Returns'].mean() * 252) / (df['Returns'][df['Returns'] < 0].std() * np.sqrt(252)),
                    "Return_on_Equity": df['Return_on_Equity'].iloc[-1],
                    "Return_on_Assets": df['Return_on_Assets'].iloc[-1],
                    "Debt_to_Equity": df['Debt_to_Equity'].iloc[-1],
                    "Current_Ratio": df['Current_Ratio'].iloc[-1],
                    "Quick_Ratio": df['Quick_Ratio'].iloc[-1]
                }
                
                # Calculate sector metrics
                sector_metrics = {
                    "Sector": df['Sector'].iloc[-1],
                    "Industry": df['Industry'].iloc[-1],
                    "Market_Cap_Rank": "Large" if df['Market_Cap'].iloc[-1] > 1e10 else "Mid" if df['Market_Cap'].iloc[-1] > 1e9 else "Small",
                    "Liquidity_Score": "High" if df['Avg_Daily_Volume'].iloc[-1] > 1e6 else "Medium" if df['Avg_Daily_Volume'].iloc[-1] > 1e5 else "Low",
                    "Gross_Margin": df['Gross_Margin'].iloc[-1],
                    "Operating_Margin": df['Operating_Margin'].iloc[-1],
                    "Net_Margin": df['Net_Margin'].iloc[-1]
                }
                
                # Add intraday-specific metrics for shorter timeframes
                if timeframe in ["1h", "15m"]:
                    intraday_metrics = {
                        "Intraday_Volatility": df['Intraday_Volatility'].iloc[-1] if 'Intraday_Volatility' in df.columns else 0,
                        "Volume_Ratio": df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else 0,
                        "Price_Momentum": df['Price_Momentum'].iloc[-1] if 'Price_Momentum' in df.columns else 0,
                        "Volume_Momentum": df['Volume_Momentum'].iloc[-1] if 'Volume_Momentum' in df.columns else 0,
                        "Volume_Price_Trend": df['Volume_Price_Trend'].iloc[-1] if 'Volume_Price_Trend' in df.columns else 0
                    }
                    product_metrics.update(intraday_metrics)
                
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
                ld (int): Historical lookback period in days (1-60)
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
                ld (int): Historical lookback period in days (1-7)
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