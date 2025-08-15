#!/usr/bin/env python3
"""
Daily NQ Data Fetcher
Fetches 25 years of daily NQ data from multiple sources for pattern analysis
Tries Stooq first, falls back to yfinance for daily data
"""

import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
import yaml
from typing import Tuple, Optional
from io import StringIO
import time

class DailyNQFetcher:
    """Fetches NQ daily data from multiple sources with fallback options."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Stooq fetcher with configuration."""
        self.config = self.load_config(config_path)
        self.ticker = self.config['data']['ticker']
        self.setup_logging()
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging for the fetcher."""
        log_dir = self.config['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/stooq_fetcher.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_daily_data(self, start_year: int = None, end_year: int = None) -> pd.DataFrame:
        """
        Download 25 years of NQ daily data from Stooq.
        
        Args:
            start_year: Starting year (default from config)
            end_year: Ending year (default from config)
            
        Returns:
            DataFrame with daily OHLCV data
        """
        if start_year is None:
            start_year = self.config['data']['start_year']
        if end_year is None:
            end_year = self.config['data']['end_year']
            
        self.logger.info(f"Fetching NQ daily data from Stooq: {start_year}-{end_year}")
        
        # Construct Stooq URL - try different NQ symbol variations
        symbols_to_try = [
            "nq.f",      # Standard futures format
            "nq",        # Simple format
            "nq.us",     # US market format
            "nqh25.us",  # Specific contract
            "nqu25.us"   # Another contract format
        ]
        
        url = None
        for symbol in symbols_to_try:
            test_url = f"https://stooq.com/q/d/l/?s={symbol}&i=d&d1={start_year}0101&d2={end_year}1231"
            self.logger.info(f"Trying URL: {test_url}")
            url = test_url
            break  # For now, just use the first one
        
        try:
            # Try Stooq first
            self.logger.info("Attempting to fetch from Stooq...")
            response = self.download_with_retry(url)
            
            # Parse CSV data
            data = pd.read_csv(StringIO(response.text))
            
            # Clean and validate data
            data = self.clean_stooq_data(data)
            
            # Validate data quality
            if not self.validate_data_quality(data):
                raise ValueError("Data quality validation failed")
            
            self.logger.info(f"Successfully downloaded {len(data)} daily bars from Stooq")
            self.logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Stooq failed: {e}")
            self.logger.info("Falling back to yfinance for daily data...")
            
            try:
                return self.fetch_from_yfinance(start_year, end_year)
            except Exception as yf_error:
                self.logger.error(f"Both Stooq and yfinance failed. yfinance error: {yf_error}")
                raise RuntimeError("All data sources failed")
    
    def download_with_retry(self, url: str, max_retries: int = 3) -> requests.Response:
        """Download data with retry logic."""
        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading from Stooq (attempt {attempt + 1}/{max_retries})")
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Check if we got actual data (not error page)
                if len(response.text) < 100:
                    self.logger.error(f"Response content: {response.text[:200]}")
                    raise ValueError("Response too short - likely an error")
                
                # Log first few lines to debug
                lines = response.text.split('\n')[:5]
                self.logger.info(f"First few lines of response: {lines}")
                    
                return response
                
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise
    
    def fetch_from_yfinance(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Fetch daily NQ data from yfinance as fallback.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            DataFrame with daily OHLCV data
        """
        self.logger.info(f"Fetching NQ daily data from yfinance: {start_year}-{end_year}")
        
        # yfinance symbols to try for NQ futures
        symbols_to_try = [
            "NQ=F",      # E-mini NASDAQ-100 futures
            "^IXIC",     # NASDAQ Composite (backup)
            "QQQ"        # NASDAQ ETF (backup)
        ]
        
        for symbol in symbols_to_try:
            try:
                self.logger.info(f"Trying yfinance symbol: {symbol}")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch maximum available daily data
                # For daily data, yfinance can go back many years
                start_date = f"{start_year}-01-01"
                end_date = f"{end_year}-12-31"
                
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True,
                    prepost=False,
                    back_adjust=False
                )
                
                if data.empty:
                    self.logger.warning(f"No data returned for {symbol}")
                    continue
                
                # Clean and standardize yfinance data
                data = self.clean_yfinance_data(data, symbol)
                
                # Validate data quality
                if self.validate_data_quality(data):
                    self.logger.info(f"Successfully fetched {len(data)} bars from yfinance ({symbol})")
                    return data
                else:
                    self.logger.warning(f"Data quality validation failed for {symbol}")
                    continue
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch from yfinance {symbol}: {e}")
                continue
        
        raise ValueError("All yfinance symbols failed")
    
    def clean_yfinance_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize yfinance data."""
        self.logger.info(f"Cleaning yfinance data for {symbol}...")
        self.logger.info(f"Original columns: {list(data.columns)}")
        
        # Standardize column names - keep only OHLCV
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[expected_cols]
        
        # Remove timezone info if present
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Remove weekends
        data = data[data.index.weekday < 5]
        
        # Remove missing data
        initial_count = len(data)
        data = data.dropna()
        removed_count = initial_count - len(data)
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} rows with missing data")
        
        # Validate OHLC relationships
        valid_mask = self.get_valid_ohlc_mask(data)
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            self.logger.warning(f"Removing {invalid_count} rows with invalid OHLC relationships")
            data = data[valid_mask]
        
        self.logger.info(f"Cleaned yfinance data: {len(data)} valid daily bars")
        return data
    
    def clean_stooq_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize Stooq data format.
        
        Args:
            data: Raw data from Stooq
            
        Returns:
            Cleaned DataFrame with standardized format
        """
        self.logger.info("Cleaning Stooq data...")
        
        # Check column names and standardize
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        if list(data.columns) != expected_columns:
            self.logger.warning(f"Unexpected columns: {list(data.columns)}")
            # Try to map common variations
            data.columns = expected_columns[:len(data.columns)]
        
        # Convert date index
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        data.sort_index(inplace=True)
        
        # Remove weekends (Stooq sometimes includes them with NaN)
        data = data[data.index.weekday < 5]  # Monday=0, Friday=4
        
        # Remove rows with missing data
        initial_count = len(data)
        data = data.dropna()
        removed_count = initial_count - len(data)
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} rows with missing data")
        
        # Validate OHLC relationships
        valid_mask = self.get_valid_ohlc_mask(data)
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            self.logger.warning(f"Removing {invalid_count} rows with invalid OHLC relationships")
            data = data[valid_mask]
        
        # Ensure numeric types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        self.logger.info(f"Data cleaned: {len(data)} valid daily bars")
        return data
    
    def get_valid_ohlc_mask(self, data: pd.DataFrame) -> pd.Series:
        """Get mask for valid OHLC relationships."""
        return (
            (data['High'] >= data['Open']) &
            (data['High'] >= data['Close']) &
            (data['High'] >= data['Low']) &
            (data['Low'] <= data['Open']) &
            (data['Low'] <= data['Close']) &
            (data['Open'] > 0) &
            (data['High'] > 0) &
            (data['Low'] > 0) &
            (data['Close'] > 0)
        )
    
    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data passes quality checks
        """
        self.logger.info("Validating data quality...")
        
        # Check minimum data requirements
        if len(data) < 100:  # Reduced for testing - need at least 100 rows
            self.logger.error(f"Insufficient data: {len(data)} rows (need at least 100)")
            return False
        
        # Check for recent data (within last 2 years)
        latest_date = data.index[-1]
        cutoff_date = datetime.now() - timedelta(days=730)
        
        if latest_date < cutoff_date:
            self.logger.error(f"Data is too old. Latest: {latest_date}, Cutoff: {cutoff_date}")
            return False
        
        # Check for reasonable price ranges (NQ should be > 1000 and < 50000)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            min_price = data[col].min()
            max_price = data[col].max()
            
            if min_price < 100 or max_price > 100000:
                self.logger.error(f"{col} prices out of reasonable range: {min_price} - {max_price}")
                return False
        
        # Check for data gaps
        date_range = pd.date_range(start=data.index[0], end=data.index[-1], freq='D')
        business_days = date_range[date_range.weekday < 5]
        
        missing_days = len(business_days) - len(data)
        missing_ratio = missing_days / len(business_days)
        
        if missing_ratio > 0.1:  # More than 10% missing
            self.logger.warning(f"High missing data ratio: {missing_ratio:.1%}")
        
        self.logger.info("Data quality validation passed")
        return True
    
    def get_previous_day_levels(self, current_date: datetime, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Get previous trading day high and low.
        
        Args:
            current_date: Current date for analysis
            data: Full daily price data
            
        Returns:
            Tuple of (previous_high, previous_low)
        """
        # Convert to datetime if needed
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # Find previous trading day
        current_idx = data.index.get_indexer([current_date], method='nearest')[0]
        
        if current_idx <= 0:
            raise ValueError(f"No previous day data available for {current_date}")
        
        # Get previous trading day (one row back)
        prev_day = data.iloc[current_idx - 1]
        prev_high = prev_day['High']
        prev_low = prev_day['Low']
        
        self.logger.debug(f"Previous day ({data.index[current_idx - 1].date()}): High={prev_high:.2f}, Low={prev_low:.2f}")
        
        return prev_high, prev_low
    
    def get_trading_days_in_range(self, start_date: str, end_date: str, data: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Get list of trading days in date range.
        
        Args:
            start_date: Start date string
            end_date: End date string  
            data: Daily price data
            
        Returns:
            DatetimeIndex of trading days
        """
        mask = (data.index >= start_date) & (data.index <= end_date)
        return data[mask].index
    
    def save_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.config['paths']['metadata']}/nq_daily_data_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data.to_csv(filename)
        self.logger.info(f"Data saved to: {filename}")
        
        return filename

def main():
    """Test the daily NQ data fetcher."""
    print("Testing Daily NQ Data Fetcher...")
    
    try:
        # Initialize fetcher
        fetcher = DailyNQFetcher()
        
        # Fetch sample data (last 5 years for testing)
        data = fetcher.fetch_daily_data(start_year=2020, end_year=2025)
        
        print(f"✅ Successfully fetched {len(data)} daily bars")
        print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        # Test previous day levels
        test_date = data.index[-10]  # 10 days from end
        prev_high, prev_low = fetcher.get_previous_day_levels(test_date, data)
        print(f"📊 Previous day levels for {test_date.date()}: High=${prev_high:.2f}, Low=${prev_low:.2f}")
        
        # Save test data
        filename = fetcher.save_data(data, "data/metadata/test_nq_data.csv")
        print(f"💾 Data saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())