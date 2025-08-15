#!/usr/bin/env python3
"""
Daily Chart Generator for NQ Pattern Analysis
Generates 30-bar daily candlestick charts with previous day high/low levels
"""

import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os
import yaml
import logging
from pathlib import Path
from typing import Tuple, Optional

class DailyChartGenerator:
    """Generates 30-bar daily charts with previous day levels."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize chart generator with configuration."""
        self.config = self.load_config(config_path)
        self.chart_config = self.config['chart']
        self.bars_per_chart = self.config['data']['bars_per_chart']
        self.setup_logging()
        
        # Ensure output directory exists
        os.makedirs(self.config['paths']['images'], exist_ok=True)
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging for the chart generator."""
        log_dir = self.config['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/chart_generator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_chart_image(self, analysis_date: datetime, data: pd.DataFrame) -> str:
        """
        Generate 30-bar daily chart with previous day levels.
        
        Args:
            analysis_date: The date being analyzed
            data: Full daily price data
            
        Returns:
            Path to saved chart image
        """
        try:
            self.logger.info(f"Generating chart for analysis date: {analysis_date.date()}")
            
            # Get 30-bar chart window and previous day levels
            chart_data, prev_high, prev_low = self.get_chart_window(analysis_date, data)
            
            # Validate chart data
            if not self.validate_chart_data(chart_data):
                raise ValueError(f"Invalid chart data for {analysis_date}")
            
            # Generate chart
            chart_path = self.create_candlestick_chart(
                chart_data, prev_high, prev_low, analysis_date
            )
            
            self.logger.info(f"Chart generated successfully: {chart_path}")
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error generating chart for {analysis_date}: {e}")
            raise
    
    def get_chart_window(self, analysis_date: datetime, data: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
        """
        Get 30-day chart window and previous day levels.
        
        Args:
            analysis_date: Date being analyzed
            data: Full daily price data
            
        Returns:
            Tuple of (chart_data, prev_high, prev_low)
        """
        # Convert to datetime if needed
        if isinstance(analysis_date, str):
            analysis_date = pd.to_datetime(analysis_date)
        
        # Find analysis date position in data
        try:
            analysis_idx = data.index.get_indexer([analysis_date], method='nearest')[0]
        except:
            # If exact date not found, find closest business day
            analysis_idx = data.index.searchsorted(analysis_date)
            if analysis_idx >= len(data):
                analysis_idx = len(data) - 1
        
        # Ensure we have enough data for chart
        if analysis_idx < self.bars_per_chart:
            raise ValueError(f"Insufficient data for chart. Need {self.bars_per_chart} bars, have {analysis_idx}")
        
        # Get 30 bars ending the day BEFORE analysis (chart shows context up to previous day)
        chart_start_idx = analysis_idx - self.bars_per_chart
        chart_end_idx = analysis_idx  # Exclude analysis day from chart
        
        chart_data = data.iloc[chart_start_idx:chart_end_idx]
        
        # Previous day levels (last bar in chart)
        prev_day = chart_data.iloc[-1]
        prev_high = float(prev_day['High'])
        prev_low = float(prev_day['Low'])
        
        self.logger.debug(f"Chart window: {chart_data.index[0].date()} to {chart_data.index[-1].date()}")
        self.logger.debug(f"Previous day ({chart_data.index[-1].date()}): High={prev_high:.2f}, Low={prev_low:.2f}")
        
        return chart_data, prev_high, prev_low
    
    def create_candlestick_chart(self, chart_data: pd.DataFrame, prev_high: float, 
                                prev_low: float, analysis_date: datetime) -> str:
        """
        Create candlestick chart with previous day levels.
        
        Args:
            chart_data: 30 bars of daily data
            prev_high: Previous day high (green line)
            prev_low: Previous day low (red line)
            analysis_date: Date being analyzed (for filename)
            
        Returns:
            Path to saved chart image
        """
        # Prepare horizontal lines for previous day levels
        hlines = {
            'hlines': [prev_high, prev_low],
            'colors': [self.chart_config['prev_high_color'], 
                      self.chart_config['prev_low_color']],
            'linestyle': '-',
            'linewidths': self.chart_config['line_width']
        }
        
        # Configure chart style
        mc = mpf.make_marketcolors(
            up='green', down='red',
            edge='inherit',
            wick={'up': 'green', 'down': 'red'}
        )
        
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            gridcolor='lightgray',
            y_on_right=False
        )
        
        # Generate filename
        date_str = analysis_date.strftime('%Y%m%d')
        filename = f"{self.config['paths']['images']}/daily_chart_{date_str}.png"
        
        # Create the chart
        plot_kwargs = {
            'type': 'candle',
            'style': style,
            'hlines': hlines,
            'volume': False,  # Disable volume to focus on price action
            'figsize': (self.chart_config['width'], self.chart_config['height']),
            'returnfig': True,
            'datetime_format': '%m/%d',
            'xrotation': 45,
            'tight_layout': True,
            'savefig': dict(
                fname=filename,
                dpi=self.chart_config['dpi'],
                bbox_inches='tight'
            )
        }
        
        # Add title if configured
        if self.chart_config.get('title', True):
            chart_start = chart_data.index[0].strftime('%m/%d/%Y')
            chart_end = chart_data.index[-1].strftime('%m/%d/%Y')
            plot_kwargs['title'] = (
                f"NQ Daily Pattern Analysis - {analysis_date.strftime('%Y-%m-%d')}\n"
                f"Chart Context: {chart_start} to {chart_end} | "
                f"Prev High (Green): {prev_high:.2f} | Prev Low (Red): {prev_low:.2f}"
            )
        
        # Create and save the chart
        fig, axes = mpf.plot(chart_data, **plot_kwargs)
        
        # Close figure to free memory
        plt.close(fig)
        
        return filename
    
    def validate_chart_data(self, chart_data: pd.DataFrame) -> bool:
        """
        Validate chart data quality.
        
        Args:
            chart_data: DataFrame to validate
            
        Returns:
            True if data is valid for charting
        """
        # Check minimum bars requirement
        if len(chart_data) < 10:  # At least 10 bars for meaningful chart
            self.logger.warning(f"Insufficient chart data: {len(chart_data)} bars")
            return False
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_columns if col not in chart_data.columns]
        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for valid OHLC relationships
        invalid_candles = (
            (chart_data['High'] < chart_data['Low']) |
            (chart_data['High'] < chart_data['Open']) |
            (chart_data['High'] < chart_data['Close']) |
            (chart_data['Low'] > chart_data['Open']) |
            (chart_data['Low'] > chart_data['Close'])
        )
        
        if invalid_candles.any():
            invalid_count = invalid_candles.sum()
            self.logger.warning(f"Found {invalid_count} invalid OHLC relationships")
            return False
        
        # Check index is datetime
        if not isinstance(chart_data.index, pd.DatetimeIndex):
            self.logger.warning("Chart data index is not DatetimeIndex")
            return False
        
        return True
    
    def add_level_annotations(self, fig, axes, prev_high: float, prev_low: float):
        """
        Add text annotations for levels on the chart.
        
        Args:
            fig: Matplotlib figure
            axes: Chart axes
            prev_high: Previous day high
            prev_low: Previous day low
        """
        if not axes:
            return
            
        ax = axes[0] if isinstance(axes, (list, tuple)) else axes
        
        # Get chart bounds
        y_range = ax.get_ylim()
        x_range = ax.get_xlim()
        
        # Position text at right edge
        x_pos = x_range[1] * 0.98
        
        # Add high label
        ax.text(x_pos, prev_high, f'PDH: {prev_high:.2f}',
                color='green', fontweight='bold', fontsize=10,
                va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add low label  
        ax.text(x_pos, prev_low, f'PDL: {prev_low:.2f}',
                color='red', fontweight='bold', fontsize=10,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def batch_generate_charts(self, data: pd.DataFrame, start_date: str = None, 
                             end_date: str = None) -> list:
        """
        Generate charts for multiple dates in batch.
        
        Args:
            data: Full daily price data
            start_date: Start date for chart generation
            end_date: End date for chart generation
            
        Returns:
            List of generated chart paths
        """
        chart_paths = []
        errors = []
        
        # Get date range for chart generation
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = data.index[self.bars_per_chart]  # Start after we have enough bars
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = data.index[-1]
        
        # Filter trading days in range
        analysis_dates = data.loc[start_dt:end_dt].index
        
        self.logger.info(f"Generating charts for {len(analysis_dates)} trading days")
        
        for i, analysis_date in enumerate(analysis_dates):
            try:
                chart_path = self.generate_chart_image(analysis_date, data)
                chart_paths.append(chart_path)
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Generated {i + 1}/{len(analysis_dates)} charts")
                    
            except Exception as e:
                error_msg = f"Error generating chart for {analysis_date}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        self.logger.info(f"Batch generation complete: {len(chart_paths)} charts, {len(errors)} errors")
        
        if errors:
            self.logger.warning(f"Errors encountered: {errors[:5]}...")  # Log first 5 errors
        
        return chart_paths

def main():
    """Test the daily chart generator."""
    print("Testing Daily Chart Generator...")
    
    try:
        # Initialize generator
        generator = DailyChartGenerator()
        
        # Load test data (from our fetcher test)
        test_data_path = "data/metadata/test_nq_data.csv"
        
        if not os.path.exists(test_data_path):
            print("❌ Test data not found. Run stooq_fetcher.py first.")
            return 1
        
        # Load data
        data = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
        print(f"📊 Loaded {len(data)} daily bars")
        
        # Test single chart generation
        test_date = data.index[-50]  # 50 days from end for testing
        chart_path = generator.generate_chart_image(test_date, data)
        
        print(f"✅ Chart generated: {chart_path}")
        print(f"📅 Analysis date: {test_date.date()}")
        
        # Test batch generation (last 10 days)
        recent_data = data.tail(60)  # Get last 60 days
        start_date = recent_data.index[-10]  # Generate for last 10 days
        
        chart_paths = generator.batch_generate_charts(
            data=data,
            start_date=start_date.strftime('%Y-%m-%d')
        )
        
        print(f"📈 Batch generated {len(chart_paths)} charts")
        print("✅ Chart generator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())