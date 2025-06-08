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

# Initialize global variables
pipeline = None
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform([[-1, 1]])

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

@spaces.GPU
def load_pipeline():
    """Load the Chronos model with GPU configuration"""
    global pipeline
    try:
        if pipeline is None:
            clear_gpu_memory()
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                device_map="gpu",  # Let the model decide the best device mapping
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            )
            pipeline.model = pipeline.model.eval()
        return pipeline
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        # Fallback to CPU if GPU fails
        if "cuda" in str(e).lower():
            print("Falling back to CPU mode")
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            pipeline.model = pipeline.model.eval()
        return pipeline

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
        # Map timeframe to yfinance interval
        tf_map = {
            "1d": "1d",
            "1h": "1h",
            "15m": "15m"
        }
        interval = tf_map.get(timeframe, "1d")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Fetch data using yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        # Get additional info for structured products
        info = ticker.info
        df['Market_Cap'] = info.get('marketCap', None)
        df['Sector'] = info.get('sector', None)
        df['Industry'] = info.get('industry', None)
        df['Dividend_Yield'] = info.get('dividendYield', None)
        
        # Calculate technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        
        # Calculate returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Annualized_Vol'] = df['Volatility'] * np.sqrt(252)  # Annualized volatility
        
        # Calculate drawdown metrics
        df['Rolling_Max'] = df['Close'].rolling(window=252, min_periods=1).max()
        df['Drawdown'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
        df['Max_Drawdown'] = df['Drawdown'].rolling(window=252, min_periods=1).min()
        
        # Calculate liquidity metrics
        df['Avg_Daily_Volume'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Volatility'] = df['Volume'].rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        raise Exception(f"Error fetching historical data for {symbol}: {str(e)}")

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and Signal line"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band, middle_band, lower_band

@spaces.GPU
def make_prediction(symbol: str, timeframe: str = "1d", prediction_days: int = 5, strategy: str = "chronos") -> Tuple[Dict, go.Figure]:
    """
    Make prediction using selected strategy.
    
    Args:
        symbol (str): Stock symbol
        timeframe (str): Data timeframe
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
                returns = df['Returns'].values
                normalized_returns = (returns - returns.mean()) / returns.std()
                context = torch.tensor(normalized_returns.reshape(-1, 1), dtype=torch.float32)
                
                # Make prediction with GPU acceleration
                pipe = load_pipeline()
                
                # Limit prediction length to avoid memory issues
                actual_prediction_days = min(prediction_days, 64)
                
                with torch.inference_mode():
                    prediction = pipe.predict(
                        context=context,
                        prediction_length=actual_prediction_days,
                        num_samples=100 
                    ).detach().cpu().numpy()
                
                mean_pred = prediction.mean(axis=0)
                std_pred = prediction.std(axis=0)
                
                # If we had to limit the prediction days, extend the prediction
                if actual_prediction_days < prediction_days:
                    last_pred = mean_pred[-1]
                    last_std = std_pred[-1]
                    extension = np.array([last_pred * (1 + np.random.normal(0, last_std, prediction_days - actual_prediction_days))])
                    mean_pred = np.concatenate([mean_pred, extension])
                    std_pred = np.concatenate([std_pred, np.full(prediction_days - actual_prediction_days, last_std)])
                
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
        
        # Create prediction dates
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
        
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
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Analysis and Prediction',
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
            "prediction": mean_pred.tolist(),
            "confidence": std_pred.tolist(),
            "dates": pred_dates.strftime('%Y-%m-%d').tolist(),
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
    """Create the Gradio interface"""
    with gr.Blocks(title="Structured Product Analysis") as demo:
        gr.Markdown("# Structured Product Analysis")
        gr.Markdown("Analyze stocks for inclusion in structured financial products with extended time horizons.")
        
        with gr.Row():
            with gr.Column():
                symbol = gr.Textbox(label="Stock Symbol (e.g., AAPL)", value="AAPL")
                timeframe = gr.Dropdown(
                    choices=["1d", "1h", "15m"],
                    label="Timeframe",
                    value="1d"
                )
                prediction_days = gr.Slider(
                    minimum=1,
                    maximum=365,  # Extended to 1 year
                    value=30,
                    step=1,
                    label="Days to Predict"
                )
                lookback_days = gr.Slider(
                    minimum=365,
                    maximum=3650,  # 10 years of history
                    value=365,
                    step=30,
                    label="Historical Lookback (Days)"
                )
                strategy = gr.Dropdown(
                    choices=["chronos", "technical"],
                    label="Prediction Strategy",
                    value="chronos"
                )
                predict_btn = gr.Button("Analyze Stock")
            
            with gr.Column():
                plot = gr.Plot(label="Analysis and Prediction")
                signals = gr.JSON(label="Trading Signals")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Structured Product Metrics")
                metrics = gr.JSON(label="Product Metrics")
                
                gr.Markdown("### Risk Analysis")
                risk_metrics = gr.JSON(label="Risk Metrics")
                
                gr.Markdown("### Sector Analysis")
                sector_metrics = gr.JSON(label="Sector Metrics")
        
        def analyze_stock(symbol, timeframe, prediction_days, lookback_days, strategy):
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
        
        predict_btn.click(
            fn=analyze_stock,
            inputs=[symbol, timeframe, prediction_days, lookback_days, strategy],
            outputs=[signals, plot, metrics, risk_metrics, sector_metrics]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True) 