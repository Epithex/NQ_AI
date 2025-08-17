#!/usr/bin/env python3
"""
Excel Data Fetcher for NQ_AI
Loads data directly from Excel file for DOW, NASDAQ, SP500 futures
Replaces API-based data fetching with local Excel data source
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import yaml
from typing import Tuple, Optional, List, Dict
from pathlib import Path


class ExcelDataFetcher:
    """Fetches daily data for multiple instruments from Excel file."""

    def __init__(self, config_path: str = "config/config_binary_visual.yaml"):
        """Initialize Excel data fetcher with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Excel file configuration
        self.excel_file = self.config["data"]["excel_file"]
        self.available_instruments = self.config["data"]["available_instruments"]
        
        # Instrument to sheet mapping from config
        self.sheet_mapping = {}
        self.display_names = {}
        
        for instrument, config in self.config["data"]["instrument_config"].items():
            self.sheet_mapping[instrument] = config["sheet"]
            self.display_names[instrument] = config["display_name"]
        
        # Column mapping for standardization
        self.column_mapping = {
            "Date": "Date",
            "Price": "Close",  # Price column maps to Close
            "Open": "Open",
            "High": "High", 
            "Low": "Low",
            "Vol.": "Volume",
            "Change %": "Change_Pct"
        }

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging for the fetcher."""
        log_dir = self.config["paths"]["logs_dir"]
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(f"{log_dir}/excel_data_fetcher.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def fetch_instrument_data(
        self, 
        instrument: str, 
        start_year: int = None, 
        end_year: int = None
    ) -> pd.DataFrame:
        """
        Fetch data for a specific instrument from Excel file.

        Args:
            instrument: Instrument name (DOW, NASDAQ, SP500)
            start_year: Start year for data (default from config)
            end_year: End year for data (default from config)

        Returns:
            DataFrame with OHLC data
        """
        if start_year is None:
            start_year = self.config["data"]["start_year"]
        if end_year is None:
            end_year = self.config["data"]["end_year"]
            
        # Validate instrument
        if instrument not in self.sheet_mapping:
            raise ValueError(f"Invalid instrument: {instrument}. Available: {list(self.sheet_mapping.keys())}")
        
        # Get sheet name
        sheet_name = self.sheet_mapping[instrument]
        
        self.logger.info(f"Loading {instrument} data from Excel sheet '{sheet_name}': {start_year}-{end_year}")
        
        try:
            # Load data from Excel
            if not os.path.exists(self.excel_file):
                raise FileNotFoundError(f"Excel file not found: {self.excel_file}")
            
            # Read the specific sheet
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            
            # Clean and standardize data
            df = self.clean_excel_data(df, instrument)
            
            # Filter by date range
            if start_year or end_year:
                df = self.filter_by_date_range(df, start_year, end_year)
            
            # Validate data quality
            if not self.validate_data_quality(df, instrument):
                raise ValueError(f"Data quality validation failed for {instrument}")
                
            self.logger.info(f"Successfully loaded {len(df)} daily bars for {instrument}")
            self.logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {instrument} data: {e}")
            raise

    def fetch_multi_instrument_data(
        self, 
        instruments: List[str], 
        start_year: int = None, 
        end_year: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple instruments.

        Args:
            instruments: List of instrument names
            start_year: Start year for data
            end_year: End year for data

        Returns:
            Dictionary mapping instrument names to DataFrames
        """
        self.logger.info(f"Loading multi-instrument data: {instruments}")
        
        results = {}
        for instrument in instruments:
            try:
                data = self.fetch_instrument_data(instrument, start_year, end_year)
                results[instrument] = data
                self.logger.info(f"✅ {instrument}: {len(data)} bars loaded")
            except Exception as e:
                self.logger.error(f"❌ Failed to load {instrument}: {e}")
                continue
        
        self.logger.info(f"Multi-instrument loading complete: {len(results)}/{len(instruments)} successful")
        return results

    def clean_excel_data(self, data: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """
        Clean and standardize Excel data format.

        Args:
            data: Raw data from Excel
            instrument: Instrument name for logging

        Returns:
            Cleaned DataFrame with standardized format
        """
        self.logger.info(f"Cleaning Excel data for {instrument}...")
        
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Rename columns to standard format
        df = df.rename(columns=self.column_mapping)
        
        # Ensure we have required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for {instrument}: {missing_columns}")
        
        # Convert date column and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove weekends (if any)
        df = df[df.index.weekday < 5]  # Monday=0, Friday=4
        
        # Remove rows with missing OHLC data
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        initial_count = len(df)
        df = df.dropna(subset=ohlc_columns)
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} rows with missing OHLC data")
        
        # Handle missing volume data (set to 0 if missing)
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        else:
            df['Volume'] = 0
            self.logger.info(f"Added default Volume column for {instrument}")
        
        # Validate OHLC relationships
        valid_mask = self.get_valid_ohlc_mask(df)
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            self.logger.warning(f"Removing {invalid_count} rows with invalid OHLC relationships")
            df = df[valid_mask]
        
        # Ensure numeric types for OHLC columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows that became NaN after numeric conversion
        df = df.dropna(subset=ohlc_columns)
        
        self.logger.info(f"Data cleaned for {instrument}: {len(df)} valid daily bars")
        return df

    def filter_by_date_range(
        self, 
        data: pd.DataFrame, 
        start_year: int, 
        end_year: int
    ) -> pd.DataFrame:
        """
        Filter data by date range.

        Args:
            data: DataFrame with date index
            start_year: Start year
            end_year: End year

        Returns:
            Filtered DataFrame
        """
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        mask = (data.index >= start_date) & (data.index <= end_date)
        filtered_data = data[mask]
        
        self.logger.info(f"Filtered data from {start_date} to {end_date}: {len(filtered_data)} bars")
        return filtered_data

    def get_valid_ohlc_mask(self, data: pd.DataFrame) -> pd.Series:
        """Get mask for valid OHLC relationships."""
        return (
            (data["High"] >= data["Open"])
            & (data["High"] >= data["Close"])
            & (data["High"] >= data["Low"])
            & (data["Low"] <= data["Open"])
            & (data["Low"] <= data["Close"])
            & (data["Open"] > 0)
            & (data["High"] > 0)
            & (data["Low"] > 0)
            & (data["Close"] > 0)
        )

    def validate_data_quality(self, data: pd.DataFrame, instrument: str) -> bool:
        """
        Validate data quality and completeness.

        Args:
            data: DataFrame to validate
            instrument: Instrument name for logging

        Returns:
            True if data passes quality checks
        """
        self.logger.info(f"Validating data quality for {instrument}...")

        # Check minimum data requirements
        if len(data) < 100:
            self.logger.error(f"Insufficient data for {instrument}: {len(data)} rows (need at least 100)")
            return False

        # Check for recent data (data should extend into recent years)
        latest_date = data.index[-1]
        if latest_date.year < 2020:  # Expect data to be relatively recent
            self.logger.warning(f"Data for {instrument} may be outdated. Latest: {latest_date}")

        # Check for reasonable price ranges
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            if col in data.columns:
                min_price = data[col].min()
                max_price = data[col].max()

                # Very broad range check (different instruments have different ranges)
                if min_price < 1 or max_price > 100000:
                    self.logger.warning(f"{instrument} {col} prices in unusual range: {min_price} - {max_price}")

        # Check for data continuity
        date_diff = data.index[1:] - data.index[:-1]
        large_gaps = date_diff[date_diff > timedelta(days=7)]  # Gaps > 1 week
        
        if len(large_gaps) > 10:
            self.logger.warning(f"{instrument} has {len(large_gaps)} large date gaps")

        self.logger.info(f"Data quality validation passed for {instrument}")
        return True

    def get_available_instruments(self) -> List[str]:
        """Get list of available instruments."""
        return self.available_instruments.copy()

    def get_available_sheets(self) -> List[str]:
        """Get list of available Excel sheets."""
        try:
            excel_data = pd.ExcelFile(self.excel_file)
            return excel_data.sheet_names
        except Exception as e:
            self.logger.error(f"Error reading Excel sheets: {e}")
            return []

    def get_data_summary(self, instrument: str) -> Dict:
        """
        Get summary information for an instrument.

        Args:
            instrument: Instrument name

        Returns:
            Dictionary with data summary
        """
        try:
            data = self.fetch_instrument_data(instrument)
            
            summary = {
                "instrument": instrument,
                "total_bars": len(data),
                "date_range": {
                    "start": data.index[0].isoformat(),
                    "end": data.index[-1].isoformat(),
                    "span_years": (data.index[-1] - data.index[0]).days / 365.25
                },
                "price_range": {
                    "min_close": float(data["Close"].min()),
                    "max_close": float(data["Close"].max()),
                    "latest_close": float(data["Close"].iloc[-1])
                },
                "data_quality": {
                    "missing_values": int(data.isnull().sum().sum()),
                    "valid_ohlc": int(self.get_valid_ohlc_mask(data).sum()),
                    "completion_rate": len(data) / ((data.index[-1] - data.index[0]).days / 7 * 5)  # Approximate business days
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting summary for {instrument}: {e}")
            return {"error": str(e)}

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['paths']['metadata']}/excel_data_{timestamp}.csv"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data.to_csv(filename)
        self.logger.info(f"Data saved to: {filename}")

        return filename


def main():
    """Test the Excel data fetcher."""
    print("Testing Excel Data Fetcher...")

    try:
        # Initialize fetcher
        fetcher = ExcelDataFetcher()

        print(f"✅ Excel Data Fetcher initialized")
        print(f"📁 Excel file: {fetcher.excel_file}")
        print(f"🎯 Available instruments: {fetcher.get_available_instruments()}")
        print(f"📋 Available sheets: {fetcher.get_available_sheets()}")

        # Test single instrument
        print(f"\n🧪 Testing single instrument (NASDAQ)...")
        nasdaq_data = fetcher.fetch_instrument_data("NASDAQ", start_year=2020, end_year=2023)
        
        print(f"✅ NASDAQ data loaded: {len(nasdaq_data)} bars")
        print(f"📅 Date range: {nasdaq_data.index[0].date()} to {nasdaq_data.index[-1].date()}")
        print(f"💰 Price range: ${nasdaq_data['Close'].min():.2f} - ${nasdaq_data['Close'].max():.2f}")

        # Test multi-instrument
        print(f"\n🧪 Testing multi-instrument loading...")
        multi_data = fetcher.fetch_multi_instrument_data(["DOW", "SP500"], start_year=2022, end_year=2023)
        
        for instrument, data in multi_data.items():
            print(f"✅ {instrument}: {len(data)} bars loaded")

        # Test data summary
        print(f"\n🧪 Testing data summary...")
        summary = fetcher.get_data_summary("SP500")
        print(f"📊 SP500 Summary:")
        print(f"   Total bars: {summary['total_bars']}")
        print(f"   Date span: {summary['date_range']['span_years']:.1f} years")
        print(f"   Price range: ${summary['price_range']['min_close']:.2f} - ${summary['price_range']['max_close']:.2f}")

        print(f"\n✅ Excel Data Fetcher test completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())