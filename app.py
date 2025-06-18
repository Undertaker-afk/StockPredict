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
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Additional imports for advanced features
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available. Regime detection will use simplified methods.")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    print("Warning: scikit-learn not available. Ensemble methods will be simplified.")

# Initialize global variables
pipeline = None
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform([[-1, 1]])

# Global market data cache
market_data_cache = {}
cache_expiry = {}
CACHE_DURATION = 3600  # 1 hour cache

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
def make_prediction(symbol: str, timeframe: str = "1d", prediction_days: int = 5, strategy: str = "chronos",
                   use_ensemble: bool = True, use_regime_detection: bool = True, use_stress_testing: bool = True,
                   risk_free_rate: float = 0.02, ensemble_weights: Dict = None, 
                   market_index: str = "^GSPC") -> Tuple[Dict, go.Figure]:
    """
    Make prediction using selected strategy with advanced features.
    
    Args:
        symbol (str): Stock symbol
        timeframe (str): Data timeframe ('1d', '1h', '15m')
        prediction_days (int): Number of days to predict
        strategy (str): Prediction strategy to use
        use_ensemble (bool): Whether to use ensemble methods
        use_regime_detection (bool): Whether to use regime detection
        use_stress_testing (bool): Whether to perform stress testing
        risk_free_rate (float): Risk-free rate for calculations
        ensemble_weights (Dict): Weights for ensemble models
        market_index (str): Market index for correlation analysis
    
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

def get_market_data(symbol: str = "^GSPC", lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch market data (S&P 500 by default) for correlation analysis and regime detection.
    
    Args:
        symbol (str): Market index symbol (default: ^GSPC for S&P 500)
        lookback_days (int): Number of days to look back
    
    Returns:
        pd.DataFrame: Market data with returns
    """
    cache_key = f"{symbol}_{lookback_days}"
    current_time = time.time()
    
    # Check cache
    if cache_key in market_data_cache and current_time < cache_expiry.get(cache_key, 0):
        return market_data_cache[cache_key]
    
    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        def fetch_market_history():
            return ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                prepost=False,
                actions=False,
                auto_adjust=True
            )
        
        df = retry_yfinance_request(fetch_market_history)
        
        if not df.empty:
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Cache the data
            market_data_cache[cache_key] = df
            cache_expiry[cache_key] = current_time + CACHE_DURATION
            
        return df
    except Exception as e:
        print(f"Warning: Could not fetch market data for {symbol}: {str(e)}")
        return pd.DataFrame()

def detect_market_regime(returns: pd.Series, n_regimes: int = 3) -> Dict:
    """
    Detect market regime using Hidden Markov Model or simplified methods.
    
    Args:
        returns (pd.Series): Price returns
        n_regimes (int): Number of regimes to detect
    
    Returns:
        Dict: Regime information including probabilities and characteristics
    """
    if len(returns) < 50:
        return {"regime": 1, "probabilities": [1.0], "volatility": returns.std()}
    
    try:
        if HMM_AVAILABLE:
            # Use HMM for regime detection
            model = hmm.GaussianHMM(n_components=n_regimes, random_state=42, covariance_type="full")
            model.fit(returns.dropna().reshape(-1, 1))
            
            # Get regime probabilities for the last observation
            regime_probs = model.predict_proba(returns.dropna().reshape(-1, 1))
            current_regime = model.predict(returns.dropna().reshape(-1, 1))[-1]
            
            # Calculate regime characteristics
            regime_means = model.means_.flatten()
            regime_vols = np.sqrt(model.covars_.diagonal(axis1=1, axis2=2))
            
            return {
                "regime": int(current_regime),
                "probabilities": regime_probs[-1].tolist(),
                "means": regime_means.tolist(),
                "volatilities": regime_vols.tolist(),
                "method": "HMM"
            }
        else:
            # Simplified regime detection using volatility clustering
            volatility = returns.rolling(window=20).std().dropna()
            vol_percentile = volatility.iloc[-1] / volatility.quantile(0.8)
            
            if vol_percentile > 1.2:
                regime = 2  # High volatility regime
            elif vol_percentile < 0.8:
                regime = 0  # Low volatility regime
            else:
                regime = 1  # Normal regime
            
            return {
                "regime": regime,
                "probabilities": [0.1, 0.8, 0.1] if regime == 1 else [0.8, 0.1, 0.1] if regime == 0 else [0.1, 0.1, 0.8],
                "volatility": volatility.iloc[-1],
                "method": "Volatility-based"
            }
    except Exception as e:
        print(f"Warning: Regime detection failed: {str(e)}")
        return {"regime": 1, "probabilities": [1.0], "volatility": returns.std(), "method": "Fallback"}

def calculate_advanced_risk_metrics(df: pd.DataFrame, market_returns: pd.Series = None, 
                                  risk_free_rate: float = 0.02) -> Dict:
    """
    Calculate advanced risk metrics including tail risk and market correlation.
    
    Args:
        df (pd.DataFrame): Stock data
        market_returns (pd.Series): Market returns for correlation analysis
        risk_free_rate (float): Annual risk-free rate
    
    Returns:
        Dict: Advanced risk metrics
    """
    returns = df['Returns'].dropna()
    
    if len(returns) < 30:
        return {"error": "Insufficient data for risk calculation"}
    
    # Basic metrics
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    
    # Market-adjusted metrics
    if market_returns is not None and len(market_returns) > 0:
        # Align dates
        aligned_returns = returns.reindex(market_returns.index).dropna()
        aligned_market = market_returns.reindex(aligned_returns.index).dropna()
        
        if len(aligned_returns) > 10:
            beta = np.cov(aligned_returns, aligned_market)[0,1] / np.var(aligned_market)
            alpha = aligned_returns.mean() - beta * aligned_market.mean()
            correlation = np.corrcoef(aligned_returns, aligned_market)[0,1]
        else:
            beta = 1.0
            alpha = 0.0
            correlation = 0.0
    else:
        beta = 1.0
        alpha = 0.0
        correlation = 0.0
    
    # Tail risk metrics
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Skewness and kurtosis
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # Risk-adjusted returns
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() > 0 else 0
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Information ratio (if market data available)
    if market_returns is not None and len(market_returns) > 0:
        excess_returns = aligned_returns - aligned_market
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    else:
        information_ratio = 0
    
    return {
        "Annual_Return": annual_return,
        "Annual_Volatility": annual_vol,
        "Sharpe_Ratio": sharpe_ratio,
        "Sortino_Ratio": sortino_ratio,
        "Calmar_Ratio": calmar_ratio,
        "Information_Ratio": information_ratio,
        "Beta": beta,
        "Alpha": alpha * 252,
        "Correlation_with_Market": correlation,
        "VaR_95": var_95,
        "VaR_99": var_99,
        "CVaR_95": cvar_95,
        "CVaR_99": cvar_99,
        "Max_Drawdown": max_drawdown,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Risk_Free_Rate": risk_free_rate
    }

def create_ensemble_prediction(df: pd.DataFrame, prediction_days: int, 
                             ensemble_weights: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create ensemble prediction combining multiple models.
    
    Args:
        df (pd.DataFrame): Historical data
        prediction_days (int): Number of days to predict
        ensemble_weights (Dict): Weights for different models
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and uncertainty predictions
    """
    if ensemble_weights is None:
        ensemble_weights = {"chronos": 0.6, "technical": 0.2, "statistical": 0.2}
    
    predictions = {}
    uncertainties = {}
    
    # Chronos prediction (placeholder - will be filled by main prediction function)
    predictions["chronos"] = np.array([])
    uncertainties["chronos"] = np.array([])
    
    # Technical prediction
    if ensemble_weights.get("technical", 0) > 0:
        try:
            last_price = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            volatility = df['Volatility'].iloc[-1]
            
            # Enhanced technical prediction
            trend = 1 if (rsi > 50 and macd > macd_signal) else -1
            mean_reversion = (df['SMA_200'].iloc[-1] - last_price) / last_price if 'SMA_200' in df.columns else 0
            
            tech_pred = []
            for i in range(1, prediction_days + 1):
                # Combine trend and mean reversion
                prediction = last_price * (1 + trend * volatility * 0.3 + mean_reversion * 0.1 * i)
                tech_pred.append(prediction)
            
            predictions["technical"] = np.array(tech_pred)
            uncertainties["technical"] = np.array([volatility * last_price * i for i in range(1, prediction_days + 1)])
        except Exception as e:
            print(f"Technical prediction error: {str(e)}")
            predictions["technical"] = np.array([])
            uncertainties["technical"] = np.array([])
    
    # Statistical prediction (ARIMA-like)
    if ensemble_weights.get("statistical", 0) > 0:
        try:
            returns = df['Returns'].dropna()
            if len(returns) > 10:
                # Simple moving average with momentum
                ma_short = df['Close'].rolling(window=10).mean().iloc[-1]
                ma_long = df['Close'].rolling(window=30).mean().iloc[-1]
                momentum = (ma_short - ma_long) / ma_long
                
                last_price = df['Close'].iloc[-1]
                stat_pred = []
                for i in range(1, prediction_days + 1):
                    # Mean reversion with momentum
                    prediction = last_price * (1 + momentum * 0.5 - 0.001 * i)  # Decay factor
                    stat_pred.append(prediction)
                
                predictions["statistical"] = np.array(stat_pred)
                uncertainties["statistical"] = np.array([returns.std() * last_price * np.sqrt(i) for i in range(1, prediction_days + 1)])
            else:
                predictions["statistical"] = np.array([])
                uncertainties["statistical"] = np.array([])
        except Exception as e:
            print(f"Statistical prediction error: {str(e)}")
            predictions["statistical"] = np.array([])
            uncertainties["statistical"] = np.array([])
    
    # Combine predictions
    valid_predictions = {k: v for k, v in predictions.items() if len(v) > 0}
    valid_uncertainties = {k: v for k, v in uncertainties.items() if len(v) > 0}
    
    if not valid_predictions:
        return np.array([]), np.array([])
    
    # Weighted ensemble
    total_weight = sum(ensemble_weights.get(k, 0) for k in valid_predictions.keys())
    if total_weight == 0:
        return np.array([]), np.array([])
    
    # Normalize weights
    normalized_weights = {k: ensemble_weights.get(k, 0) / total_weight for k in valid_predictions.keys()}
    
    # Calculate weighted mean and uncertainty
    max_length = max(len(v) for v in valid_predictions.values())
    ensemble_mean = np.zeros(max_length)
    ensemble_uncertainty = np.zeros(max_length)
    
    for model, pred in valid_predictions.items():
        weight = normalized_weights[model]
        if len(pred) < max_length:
            # Extend prediction using last value
            extended_pred = np.concatenate([pred, np.full(max_length - len(pred), pred[-1])])
            extended_unc = np.concatenate([valid_uncertainties[model], np.full(max_length - len(pred), valid_uncertainties[model][-1])])
        else:
            extended_pred = pred[:max_length]
            extended_unc = valid_uncertainties[model][:max_length]
        
        ensemble_mean += weight * extended_pred
        ensemble_uncertainty += weight * extended_unc
    
    return ensemble_mean, ensemble_uncertainty

def stress_test_scenarios(df: pd.DataFrame, prediction: np.ndarray, 
                         scenarios: Dict = None) -> Dict:
    """
    Perform stress testing under various market scenarios.
    
    Args:
        df (pd.DataFrame): Historical data
        prediction (np.ndarray): Base prediction
        scenarios (Dict): Stress test scenarios
    
    Returns:
        Dict: Stress test results
    """
    if scenarios is None:
        scenarios = {
            "market_crash": {"volatility_multiplier": 3.0, "return_shock": -0.15},
            "high_volatility": {"volatility_multiplier": 2.0, "return_shock": -0.05},
            "low_volatility": {"volatility_multiplier": 0.5, "return_shock": 0.02},
            "bull_market": {"volatility_multiplier": 1.2, "return_shock": 0.10},
            "interest_rate_shock": {"volatility_multiplier": 1.5, "return_shock": -0.08}
        }
    
    base_volatility = df['Volatility'].iloc[-1]
    base_return = df['Returns'].mean()
    last_price = df['Close'].iloc[-1]
    
    stress_results = {}
    
    for scenario_name, params in scenarios.items():
        try:
            # Calculate stressed parameters
            stressed_vol = base_volatility * params["volatility_multiplier"]
            stressed_return = base_return + params["return_shock"]
            
            # Generate stressed prediction
            stressed_pred = []
            for i, pred in enumerate(prediction):
                # Apply stress factors
                stress_factor = 1 + stressed_return * (i + 1) / 252
                volatility_impact = np.random.normal(0, stressed_vol * np.sqrt((i + 1) / 252))
                stressed_price = pred * stress_factor * (1 + volatility_impact)
                stressed_pred.append(stressed_price)
            
            # Calculate stress metrics
            stress_results[scenario_name] = {
                "prediction": np.array(stressed_pred),
                "max_loss": min(stressed_pred) / last_price - 1,
                "volatility": stressed_vol,
                "expected_return": stressed_return,
                "var_95": np.percentile([p / last_price - 1 for p in stressed_pred], 5)
            }
        except Exception as e:
            print(f"Stress test error for {scenario_name}: {str(e)}")
            stress_results[scenario_name] = {"error": str(e)}
    
    return stress_results

def calculate_skewed_uncertainty(quantiles: np.ndarray, confidence_level: float = 0.9) -> np.ndarray:
    """
    Calculate uncertainty accounting for skewness in return distributions.
    
    Args:
        quantiles (np.ndarray): Quantile predictions from Chronos
        confidence_level (float): Confidence level for uncertainty calculation
    
    Returns:
        np.ndarray: Uncertainty estimates
    """
    try:
        lower = quantiles[0, :, 0]
        median = quantiles[0, :, 1]
        upper = quantiles[0, :, 2]
        
        # Calculate skewness for each prediction point
        uncertainties = []
        for i in range(len(lower)):
            # Calculate skewness
            if upper[i] != median[i] and median[i] != lower[i]:
                skewness = (median[i] - lower[i]) / (upper[i] - median[i])
            else:
                skewness = 1.0
            
            # Adjust z-score based on skewness
            if skewness > 1.2:  # Right-skewed
                z_score = stats.norm.ppf(confidence_level) * (1 + 0.1 * skewness)
            elif skewness < 0.8:  # Left-skewed
                z_score = stats.norm.ppf(confidence_level) * (1 - 0.1 * abs(skewness))
            else:
                z_score = stats.norm.ppf(confidence_level)
            
            # Calculate uncertainty
            uncertainty = (upper[i] - lower[i]) / (2 * z_score)
            uncertainties.append(uncertainty)
        
        return np.array(uncertainties)
    except Exception as e:
        print(f"Skewed uncertainty calculation error: {str(e)}")
        # Fallback to simple calculation
        return (quantiles[0, :, 2] - quantiles[0, :, 0]) / (2 * 1.645)

def adaptive_smoothing(new_pred: np.ndarray, historical_pred: np.ndarray, 
                      prediction_uncertainty: np.ndarray) -> np.ndarray:
    """
    Apply adaptive smoothing based on prediction uncertainty.
    
    Args:
        new_pred (np.ndarray): New predictions
        historical_pred (np.ndarray): Historical predictions
        prediction_uncertainty (np.ndarray): Prediction uncertainty
    
    Returns:
        np.ndarray: Smoothed predictions
    """
    try:
        if len(historical_pred) == 0:
            return new_pred
        
        # Calculate adaptive alpha based on uncertainty
        uncertainty_ratio = prediction_uncertainty / np.mean(np.abs(historical_pred))
        
        if uncertainty_ratio > 0.1:  # High uncertainty
            alpha = 0.1  # More smoothing
        elif uncertainty_ratio < 0.05:  # Low uncertainty
            alpha = 0.5  # Less smoothing
        else:
            alpha = 0.3  # Default
        
        # Apply weighted smoothing
        smoothed = alpha * new_pred + (1 - alpha) * historical_pred[-len(new_pred):]
        return smoothed
    except Exception as e:
        print(f"Adaptive smoothing error: {str(e)}")
        return new_pred

def advanced_trading_signals(df: pd.DataFrame, regime_info: Dict = None) -> Dict:
    """
    Generate advanced trading signals with confidence levels and regime awareness.
    
    Args:
        df (pd.DataFrame): Stock data
        regime_info (Dict): Market regime information
    
    Returns:
        Dict: Advanced trading signals
    """
    try:
        # Calculate signal strength and confidence
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        
        rsi_strength = abs(rsi - 50) / 50  # 0-1 scale
        macd_strength = abs(macd - macd_signal) / df['Close'].iloc[-1]
        
        # Regime-adjusted thresholds
        if regime_info and "volatilities" in regime_info:
            volatility_regime = df['Volatility'].iloc[-1] / np.mean(regime_info["volatilities"])
        else:
            volatility_regime = 1.0
        
        # Adjust RSI thresholds based on volatility
        rsi_oversold = 30 + (volatility_regime - 1) * 10
        rsi_overbought = 70 - (volatility_regime - 1) * 10
        
        # Calculate signals with confidence
        signals = {}
        
        # RSI signal
        if rsi < rsi_oversold:
            rsi_signal = "Oversold"
            rsi_confidence = min(0.9, 0.5 + rsi_strength * 0.4)
        elif rsi > rsi_overbought:
            rsi_signal = "Overbought"
            rsi_confidence = min(0.9, 0.5 + rsi_strength * 0.4)
        else:
            rsi_signal = "Neutral"
            rsi_confidence = 0.3
        
        signals["RSI"] = {
            "signal": rsi_signal,
            "strength": rsi_strength,
            "confidence": rsi_confidence,
            "value": rsi
        }
        
        # MACD signal
        if macd > macd_signal:
            macd_signal = "Buy"
            macd_confidence = min(0.8, 0.4 + macd_strength * 40)
        else:
            macd_signal = "Sell"
            macd_confidence = min(0.8, 0.4 + macd_strength * 40)
        
        signals["MACD"] = {
            "signal": macd_signal,
            "strength": macd_strength,
            "confidence": macd_confidence,
            "value": macd
        }
        
        # Bollinger Bands signal
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            current_price = df['Close'].iloc[-1]
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            
            if current_price < bb_lower:
                bb_signal = "Buy"
                bb_confidence = 0.7
            elif current_price > bb_upper:
                bb_signal = "Sell"
                bb_confidence = 0.7
            else:
                bb_signal = "Hold"
                bb_confidence = 0.5
            
            signals["Bollinger"] = {
                "signal": bb_signal,
                "confidence": bb_confidence,
                "position": (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            }
        
        # SMA signal
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            
            if sma_20 > sma_50:
                sma_signal = "Buy"
                sma_confidence = 0.6
            else:
                sma_signal = "Sell"
                sma_confidence = 0.6
            
            signals["SMA"] = {
                "signal": sma_signal,
                "confidence": sma_confidence,
                "ratio": sma_20 / sma_50 if sma_50 != 0 else 1.0
            }
        
        # Calculate weighted overall signal
        buy_signals = []
        sell_signals = []
        
        for signal_name, signal_data in signals.items():
            if signal_data["signal"] == "Buy":
                buy_signals.append(signal_data["strength"] * signal_data["confidence"])
            elif signal_data["signal"] == "Sell":
                sell_signals.append(signal_data["strength"] * signal_data["confidence"])
        
        weighted_buy = sum(buy_signals) if buy_signals else 0
        weighted_sell = sum(sell_signals) if sell_signals else 0
        
        if weighted_buy > weighted_sell:
            overall_signal = "Buy"
            overall_confidence = weighted_buy / (weighted_buy + weighted_sell) if (weighted_buy + weighted_sell) > 0 else 0
        elif weighted_sell > weighted_buy:
            overall_signal = "Sell"
            overall_confidence = weighted_sell / (weighted_buy + weighted_sell) if (weighted_buy + weighted_sell) > 0 else 0
        else:
            overall_signal = "Hold"
            overall_confidence = 0.5
        
        return {
            "signals": signals,
            "overall_signal": overall_signal,
            "confidence": overall_confidence,
            "regime_adjusted": regime_info is not None
        }
    
    except Exception as e:
        print(f"Advanced trading signals error: {str(e)}")
        return {"error": str(e)}

def create_interface():
    """Create the Gradio interface with separate tabs for different timeframes"""
    with gr.Blocks(title="Advanced Stock Prediction Analysis") as demo:
        gr.Markdown("# Advanced Stock Prediction Analysis")
        gr.Markdown("Analyze stocks with advanced features including regime detection, ensemble methods, and stress testing.")
        
        # Add market status message
        market_status = "Market is currently closed" if not is_market_open() else "Market is currently open"
        next_trading_day = get_next_trading_day()
        gr.Markdown(f"""
        ### Market Status: {market_status}
        Next trading day: {next_trading_day.strftime('%Y-%m-%d')}
        """)
        
        # Advanced Settings Accordion
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                with gr.Column():
                    use_ensemble = gr.Checkbox(label="Use Ensemble Methods", value=True)
                    use_regime_detection = gr.Checkbox(label="Use Regime Detection", value=True)
                    use_stress_testing = gr.Checkbox(label="Use Stress Testing", value=True)
                    risk_free_rate = gr.Slider(
                        minimum=0.0,
                        maximum=0.1,
                        value=0.02,
                        step=0.001,
                        label="Risk-Free Rate (Annual)"
                    )
                    market_index = gr.Dropdown(
                        choices=["^GSPC", "^DJI", "^IXIC", "^RUT"],
                        label="Market Index for Correlation",
                        value="^GSPC"
                    )
                
                with gr.Column():
                    gr.Markdown("### Ensemble Weights")
                    chronos_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.1,
                        label="Chronos Weight"
                    )
                    technical_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        label="Technical Weight"
                    )
                    statistical_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        label="Statistical Weight"
                    )
        
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
                        
                        gr.Markdown("### Advanced Risk Analysis")
                        daily_risk_metrics = gr.JSON(label="Risk Metrics")
                        
                        gr.Markdown("### Market Regime Analysis")
                        daily_regime_metrics = gr.JSON(label="Regime Metrics")
                        
                        gr.Markdown("### Trading Signals")
                        daily_signals = gr.JSON(label="Trading Signals")
                    
                    with gr.Column():
                        gr.Markdown("### Stress Test Results")
                        daily_stress_results = gr.JSON(label="Stress Test Results")
                        
                        gr.Markdown("### Ensemble Analysis")
                        daily_ensemble_metrics = gr.JSON(label="Ensemble Metrics")
            
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
        
        def analyze_stock(symbol, timeframe, prediction_days, lookback_days, strategy,
                         use_ensemble, use_regime_detection, use_stress_testing,
                         risk_free_rate, market_index, chronos_weight, technical_weight, statistical_weight):
            try:
                # Create ensemble weights
                ensemble_weights = {
                    "chronos": chronos_weight,
                    "technical": technical_weight,
                    "statistical": statistical_weight
                }
                
                # Get market data for correlation analysis
                market_df = get_market_data(market_index, lookback_days)
                market_returns = market_df['Returns'] if not market_df.empty else None
                
                # Make prediction with advanced features
                signals, fig = make_prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    prediction_days=prediction_days,
                    strategy=strategy,
                    use_ensemble=use_ensemble,
                    use_regime_detection=use_regime_detection,
                    use_stress_testing=use_stress_testing,
                    risk_free_rate=risk_free_rate,
                    ensemble_weights=ensemble_weights,
                    market_index=market_index
                )
                
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
                
                # Calculate advanced risk metrics
                risk_metrics = calculate_advanced_risk_metrics(df, market_returns, risk_free_rate)
                
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
                
                # Extract regime and stress test information
                regime_metrics = signals.get("regime_info", {})
                stress_results = signals.get("stress_test_results", {})
                ensemble_metrics = {
                    "ensemble_used": signals.get("ensemble_used", False),
                    "ensemble_weights": ensemble_weights
                }
                
                return signals, fig, product_metrics, risk_metrics, sector_metrics, regime_metrics, stress_results, ensemble_metrics
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
        def daily_analysis(s: str, pd: int, ld: int, st: str, ue: bool, urd: bool, ust: bool,
                          rfr: float, mi: str, cw: float, tw: float, sw: float) -> Tuple[Dict, go.Figure, Dict, Dict, Dict, Dict, Dict, Dict]:
            """
            Process daily timeframe stock analysis with advanced features.

            Args:
                s (str): Stock symbol (e.g., "AAPL", "MSFT", "GOOGL")
                pd (int): Number of days to predict (1-365)
                ld (int): Historical lookback period in days (1-3650)
                st (str): Prediction strategy to use ("chronos" or "technical")
                ue (bool): Use ensemble methods
                urd (bool): Use regime detection
                ust (bool): Use stress testing
                rfr (float): Risk-free rate
                mi (str): Market index
                cw (float): Chronos weight
                tw (float): Technical weight
                sw (float): Statistical weight

            Returns:
                Tuple containing all analysis results
            """
            return analyze_stock(s, "1d", pd, ld, st, ue, urd, ust, rfr, mi, cw, tw, sw)

        daily_predict_btn.click(
            fn=daily_analysis,
            inputs=[daily_symbol, daily_prediction_days, daily_lookback_days, daily_strategy,
                   use_ensemble, use_regime_detection, use_stress_testing, risk_free_rate, market_index,
                   chronos_weight, technical_weight, statistical_weight],
            outputs=[daily_signals, daily_plot, daily_metrics, daily_risk_metrics, daily_sector_metrics,
                    daily_regime_metrics, daily_stress_results, daily_ensemble_metrics]
        )
        
        # Hourly analysis button click
        def hourly_analysis(s: str, pd: int, ld: int, st: str, ue: bool, urd: bool, ust: bool,
                           rfr: float, mi: str, cw: float, tw: float, sw: float) -> Tuple[Dict, go.Figure, Dict, Dict, Dict]:
            """
            Process hourly timeframe stock analysis with advanced features.

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