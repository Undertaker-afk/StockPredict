import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import torch
from chronos import BaseChronosPipeline
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize Chronos pipeline
pipeline = None

def load_pipeline():
    """Load the Chronos model with CPU configuration"""
    global pipeline
    if pipeline is None:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map="cpu",  # Force CPU usage
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        pipeline.model = pipeline.model.eval()
    return pipeline

def get_historical_data(symbol: str, timeframe: str = "1d") -> np.ndarray:
    """
    Fetch historical data using yfinance.
    
    Args:
        symbol (str): The stock symbol (e.g., 'AAPL')
        timeframe (str): The timeframe for data ('1d', '1h', '15m')
    
    Returns:
        np.ndarray: Array of historical prices for Chronos model
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
        if timeframe == "1d":
            start_date = end_date - timedelta(days=365)  # 1 year of daily data
        elif timeframe == "1h":
            start_date = end_date - timedelta(days=30)   # 30 days of hourly data
        else:  # 15m
            start_date = end_date - timedelta(days=7)    # 7 days of 15-min data
        
        # Fetch data using yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Drop NaN values
        df = df.dropna()
        
        # Normalize the data
        returns = df['returns'].values
        normalized_returns = (returns - returns.mean()) / returns.std()
        
        # Convert to the format expected by Chronos
        return normalized_returns.reshape(-1, 1)
        
    except Exception as e:
        raise Exception(f"Error fetching historical data for {symbol}: {str(e)}")

def make_prediction(symbol: str, timeframe: str = "1d", prediction_days: int = 5):
    """
    Make prediction using Chronos model.
    
    Args:
        symbol (str): Stock symbol
        timeframe (str): Data timeframe
        prediction_days (int): Number of days to predict
    
    Returns:
        dict: Prediction results and visualization
    """
    try:
        # Load pipeline
        pipe = load_pipeline()
        
        # Get historical data
        historical_data = get_historical_data(symbol, timeframe)
        
        # Convert to tensor
        context = torch.tensor(historical_data, dtype=torch.float32)
        
        # Make prediction
        with torch.inference_mode():
            prediction = pipe.predict(
                context=context,
                prediction_length=prediction_days,
                num_samples=100
            ).detach().cpu().numpy()
        
        # Get actual historical prices for plotting
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(period="1mo")
        
        # Create prediction dates
        last_date = hist_data.index[-1]
        pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
        
        # Calculate prediction statistics
        mean_pred = prediction.mean(axis=0)
        std_pred = prediction.std(axis=0)
        
        # Create visualization
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=('Price Prediction', 'Confidence Interval'))
        
        # Add historical data
        fig.add_trace(
            go.Scatter(x=hist_data.index, y=hist_data['Close'], name='Historical Price',
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
        
        # Add confidence interval plot
        fig.add_trace(
            go.Scatter(x=pred_dates, y=std_pred, name='Prediction Uncertainty',
                      line=dict(color='green')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            showlegend=True
        )
        
        return {
            "symbol": symbol,
            "prediction": mean_pred.tolist(),
            "confidence": std_pred.tolist(),
            "dates": pred_dates.strftime('%Y-%m-%d').tolist(),
            "plot": fig
        }
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Stock Price Prediction with Amazon Chronos") as demo:
        gr.Markdown("# Stock Price Prediction with Amazon Chronos")
        gr.Markdown("Enter a stock symbol and select prediction parameters to get price forecasts.")
        
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
                    maximum=30,
                    value=5,
                    step=1,
                    label="Days to Predict"
                )
                predict_btn = gr.Button("Make Prediction")
            
            with gr.Column():
                plot = gr.Plot(label="Prediction Visualization")
                results = gr.JSON(label="Prediction Results")
        
        predict_btn.click(
            fn=make_prediction,
            inputs=[symbol, timeframe, prediction_days],
            outputs=[results, plot]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True) 