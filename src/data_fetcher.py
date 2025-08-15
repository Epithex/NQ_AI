# src/data_fetcher.py
import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Tuple, Optional
import yaml
import logging

class DataFetcher:
    """Fetches and processes NQ futures data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataFetcher with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ticker = self.config['data']['ticker']
        self.interval = self.config['data']['interval']
        self.timezone = pytz.timezone(self.config['time']['timezone'])
        
        # Setup logging
        logging.basicConfig(
            filename=f"{self.config['paths']['logs']}/data_fetcher.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical 1-hour NQ futures data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching data for {self.ticker} from {start_date} to {end_date}")
            
            # Add buffer for 300 bars before start_date
            start_dt = pd.to_datetime(start_date)
            buffer_start = start_dt - timedelta(days=20)  # ~480 hours buffer
            
            # Fetch data
            ticker = yf.Ticker(self.ticker)
            data = ticker.history(
                start=buffer_start.strftime('%Y-%m-%d'),
                end=end_date,
                interval=self.interval
            )
            
            # Validate data
            if data.empty:
                self.logger.warning(f"No data retrieved for {self.ticker} from {start_date} to {end_date}")
                # Try with a more recent date range
                recent_start = (pd.to_datetime(end_date) - timedelta(days=700)).strftime('%Y-%m-%d')
                self.logger.info(f"Trying with more recent date range: {recent_start} to {end_date}")
                
                data = ticker.history(
                    start=recent_start,
                    end=end_date,
                    interval=self.interval
                )
                
                if data.empty:
                    raise ValueError(f"No data retrieved for {self.ticker} even with recent date range")
            
            # Convert to EST timezone
            if hasattr(data.index, 'tz_convert'):
                data.index = data.index.tz_convert(self.timezone)
            else:
                self.logger.warning("Data index is not timezone-aware, attempting to localize")
                data.index = pd.to_datetime(data.index).tz_localize('UTC').tz_convert(self.timezone)
            
            self.logger.info(f"Successfully fetched {len(data)} bars from {data.index[0]} to {data.index[-1]}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def get_previous_day_levels(self, current_date: datetime, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate previous trading day's high and low.
        
        Args:
            current_date: Current date for analysis
            data: Historical price data
            
        Returns:
            Tuple of (previous_high, previous_low)
        """
        try:
            # Get previous trading day (skip weekends)
            prev_date = current_date - timedelta(days=1)
            while prev_date.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
                prev_date -= timedelta(days=1)
            
            # Ensure timezone-aware dates
            if prev_date.tzinfo is None:
                prev_date = self.timezone.localize(prev_date)
            
            # Filter data for previous day (midnight to midnight EST)
            prev_day_start = prev_date.replace(hour=0, minute=0, second=0)
            prev_day_end = prev_date.replace(hour=23, minute=59, second=59)
            
            prev_day_data = data.loc[prev_day_start:prev_day_end]
            
            if prev_day_data.empty:
                raise ValueError(f"No data for previous day: {prev_date.date()}")
            
            prev_high = prev_day_data['High'].max()
            prev_low = prev_day_data['Low'].min()
            
            self.logger.info(f"Previous day ({prev_date.date()}): High={prev_high:.2f}, Low={prev_low:.2f}")
            return prev_high, prev_low
            
        except Exception as e:
            self.logger.error(f"Error calculating previous day levels: {str(e)}")
            raise
    
    def get_data_for_chart(self, analysis_time: datetime, data: pd.DataFrame, bars: int = 300) -> pd.DataFrame:
        """
        Get 300 bars of data ending at analysis time.
        
        Args:
            analysis_time: Timestamp for chart generation (7 AM EST)
            data: Full historical data
            bars: Number of bars to include (default 300)
            
        Returns:
            DataFrame with 300 bars for chart
        """
        try:
            # Find the index position of analysis_time
            analysis_idx = data.index.get_indexer([analysis_time], method='nearest')[0]
            
            # Get 300 bars ending at analysis time
            start_idx = max(0, analysis_idx - bars + 1)
            chart_data = data.iloc[start_idx:analysis_idx + 1]
            
            if len(chart_data) < bars:
                self.logger.warning(f"Only {len(chart_data)} bars available, requested {bars}")
            
            return chart_data
            
        except Exception as e:
            self.logger.error(f"Error getting chart data: {str(e)}")
            raise
    
    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data passes quality checks
        """
        # Check for NaN values
        if data.isnull().any().any():
            self.logger.warning("Data contains NaN values")
            return False
        
        # Check for zero or negative prices
        if (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            self.logger.warning("Data contains invalid prices")
            return False
        
        # Check OHLC relationships
        invalid_candles = (data['High'] < data['Low']) | \
                         (data['High'] < data['Open']) | \
                         (data['High'] < data['Close']) | \
                         (data['Low'] > data['Open']) | \
                         (data['Low'] > data['Close'])
        
        if invalid_candles.any():
            self.logger.warning("Data contains invalid OHLC relationships")
            return False
        
        return True