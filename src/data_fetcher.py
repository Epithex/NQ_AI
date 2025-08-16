# src/data_fetcher.py
import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
import yaml
import logging
import os

class MultiInstrumentDataFetcher:
    """Fetches and processes data for multiple futures instruments (NQ, ES, YM)."""
    
    def __init__(self, config_path: str = "config/config_pure_visual.yaml"):
        """Initialize DataFetcher with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.instruments = self.config['data']['instruments']
        self.source = self.config['data']['source']
        self.fallback_source = self.config['data'].get('fallback_source', 'yfinance')
        self.start_year = self.config['data']['start_year']
        self.end_year = self.config['data']['end_year']
        
        # Setup logging
        log_dir = self.config['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/multi_instrument_fetcher.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_instrument_data(self, instrument: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical daily data for a single instrument.
        
        Args:
            instrument: Instrument ticker (e.g., 'NQ.F', 'ES.F', 'YM.F')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with daily OHLCV data
        """
        try:
            self.logger.info(f"Fetching data for {instrument} from {start_date} to {end_date}")
            
            # Try primary source first (stooq)
            data = None
            if self.source == "stooq":
                try:
                    import pandas_datareader as pdr
                    # Convert futures ticker format for stooq
                    stooq_ticker = instrument.replace('.F', '')  # NQ.F -> NQ
                    data = pdr.get_data_stooq(stooq_ticker, start_date, end_date)
                    if not data.empty:
                        self.logger.info(f"Successfully fetched {len(data)} bars from stooq")
                except Exception as e:
                    self.logger.warning(f"Stooq fetch failed: {str(e)}, trying fallback")
            
            # Fallback to yfinance if stooq fails or not primary
            if data is None or data.empty:
                self.logger.info(f"Using yfinance for {instrument}")
                ticker = yf.Ticker(instrument)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1d"  # Daily data
                )
                
                if data.empty:
                    raise ValueError(f"No data retrieved for {instrument} from either source")
            
            # Standardize column names
            if 'Adj Close' in data.columns:
                data = data.drop('Adj Close', axis=1)
            
            # Ensure we have standard OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                self.logger.warning(f"Missing required columns for {instrument}")
                
            self.logger.info(f"Successfully fetched {len(data)} daily bars for {instrument}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {instrument}: {str(e)}")
            raise

    def fetch_multi_instrument_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all configured instruments.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (optional, uses config default)
            end_date: End date in YYYY-MM-DD format (optional, uses config default)
            
        Returns:
            Dictionary mapping instrument -> DataFrame
        """
        if start_date is None:
            start_date = f"{self.start_year}-01-01"
        if end_date is None:
            end_date = f"{self.end_year}-12-31"
            
        all_data = {}
        
        for instrument in self.instruments:
            try:
                self.logger.info(f"Fetching data for {instrument}")
                data = self.fetch_instrument_data(instrument, start_date, end_date)
                
                # Validate data quality
                if self.validate_data_quality(data):
                    all_data[instrument] = data
                    self.logger.info(f"✅ {instrument}: {len(data)} samples")
                else:
                    self.logger.warning(f"❌ {instrument}: Failed data quality checks")
                    
            except Exception as e:
                self.logger.error(f"❌ {instrument}: Failed to fetch data - {str(e)}")
                continue
                
        self.logger.info(f"Successfully fetched data for {len(all_data)} instruments")
        return all_data
    
    def get_previous_day_levels(self, current_date: pd.Timestamp, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate previous trading day's high and low for daily data.
        
        Args:
            current_date: Current date for analysis (pd.Timestamp)
            data: Historical daily price data with date index
            
        Returns:
            Tuple of (previous_high, previous_low)
        """
        try:
            # Ensure current_date is a pandas Timestamp
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            
            # Get the date of the current trading day
            current_date_only = current_date.date()
            
            # Find previous trading day in the data
            available_dates = data.index.date
            current_idx = None
            
            # Find current date in data
            for i, date in enumerate(available_dates):
                if date == current_date_only:
                    current_idx = i
                    break
            
            if current_idx is None or current_idx == 0:
                raise ValueError(f"Cannot find previous day for {current_date_only}")
            
            # Get previous day's data (previous index in the DataFrame)
            prev_day_data = data.iloc[current_idx - 1]
            
            prev_high = float(prev_day_data['High'])
            prev_low = float(prev_day_data['Low'])
            prev_date = data.index[current_idx - 1].date()
            
            self.logger.debug(f"Previous day ({prev_date}): High={prev_high:.2f}, Low={prev_low:.2f}")
            return prev_high, prev_low
            
        except Exception as e:
            self.logger.error(f"Error calculating previous day levels: {str(e)}")
            raise
    
    def get_data_for_chart(self, current_date: pd.Timestamp, data: pd.DataFrame, bars: int = 30) -> pd.DataFrame:
        """
        Get specified number of daily bars ending at current date for chart generation.
        
        Args:
            current_date: Current date for analysis
            data: Full historical daily data
            bars: Number of daily bars to include (default 30)
            
        Returns:
            DataFrame with specified bars for chart
        """
        try:
            # Ensure current_date is a pandas Timestamp
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            
            # Find the index position of current_date
            current_date_only = current_date.date()
            available_dates = data.index.date
            current_idx = None
            
            for i, date in enumerate(available_dates):
                if date == current_date_only:
                    current_idx = i
                    break
            
            if current_idx is None:
                raise ValueError(f"Date {current_date_only} not found in data")
            
            # Get specified number of bars ending at current date (inclusive)
            start_idx = max(0, current_idx - bars + 1)
            chart_data = data.iloc[start_idx:current_idx + 1]
            
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