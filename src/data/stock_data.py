import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data for a given symbol using yfinance with authentication.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        period (str): Time period to fetch (default: '1y')
        
    Returns:
        pd.DataFrame: DataFrame containing historical stock data
    """
    try:
        # Create a Ticker object with authentication
        ticker = yf.Ticker(symbol)
        
        # Set authentication credentials
        ticker._session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Fetch historical data
        df = ticker.history(period=period)
        
        if df.empty:
            logger.warning(f"No data found for symbol {symbol}")
            return pd.DataFrame()
            
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Convert date to string format
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        logger.info(f"Successfully fetched data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame() 