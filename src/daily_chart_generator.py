#!/usr/bin/env python3
"""
Pure Visual Chart Generator for Multi-Instrument Pattern Analysis
Generates clean 30-bar daily candlestick charts with previous day high/low levels
Optimized for pure visual learning with ViT models
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
import numpy as np

class PureVisualChartGenerator:
    """Generates clean 30-bar daily charts optimized for pure visual learning."""
    
    def __init__(self, config_path: str = "config/config_pure_visual.yaml"):
        """Initialize chart generator with configuration."""
        self.config = self.load_config(config_path)
        self.chart_config = self.config['chart']
        self.bars_per_chart = self.config['data']['bars_per_chart']
        self.image_size = self.chart_config['image_size']
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
    
    def generate_chart_image(self, analysis_date: datetime, data: pd.DataFrame, 
                            instrument: str = "NQ") -> str:
        """
        Generate pure visual 30-bar daily chart with previous day levels.
        
        Args:
            analysis_date: The date being analyzed
            data: Full daily price data
            instrument: Instrument identifier (NQ, ES, YM)
            
        Returns:
            Path to saved chart image
        """
        try:
            self.logger.info(f"Generating pure visual chart for {instrument} on {analysis_date.date()}")
            
            # Get 30-bar chart window and previous day levels
            chart_data, prev_high, prev_low = self.get_chart_window(analysis_date, data)
            
            # Validate chart data
            if not self.validate_chart_data(chart_data):
                raise ValueError(f"Invalid chart data for {analysis_date}")
            
            # Generate pure visual chart (224x224 for ViT)
            chart_path = self.create_pure_visual_chart(
                chart_data, prev_high, prev_low, analysis_date, instrument
            )
            
            self.logger.info(f"Pure visual chart generated: {chart_path}")
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
    
    def create_pure_visual_chart(self, chart_data: pd.DataFrame, prev_high: float, 
                                prev_low: float, analysis_date: datetime, instrument: str) -> str:
        """
        Create clean visual chart optimized for ViT model input (224x224).
        
        Args:
            chart_data: 30 bars of daily data
            prev_high: Previous day high (green line)
            prev_low: Previous day low (red line)
            analysis_date: Date being analyzed (for filename)
            instrument: Instrument identifier (NQ, ES, YM)
            
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
        
        # Configure clean chart style for pure visual learning
        mc = mpf.make_marketcolors(
            up='#00AA00', down='#FF0000',  # High contrast colors
            edge='inherit',
            wick={'up': '#00AA00', 'down': '#FF0000'}
        )
        
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='',  # No grid for cleaner visuals
            gridcolor='none',
            y_on_right=False,
            rc={'axes.linewidth': 0.5}  # Thin axes
        )
        
        # Generate filename with instrument prefix
        date_str = analysis_date.strftime('%Y%m%d')
        filename = f"{self.config['paths']['images']}/pure_visual_{instrument}_{date_str}.png"
        
        # Calculate figure size to get exactly 224x224 pixels
        dpi = self.chart_config['dpi']
        fig_size = self.image_size / dpi  # 224/100 = 2.24 inches
        
        # Create the pure visual chart
        plot_kwargs = {
            'type': 'candle',
            'style': style,
            'hlines': hlines,
            'volume': False,  # Pure price action only
            'figsize': (fig_size, fig_size),  # Square aspect ratio
            'returnfig': True,
            'title': '',  # No title for clean visuals
            'ylabel': '',  # No Y-axis label
            'xlabel': '',  # No X-axis label
            'datetime_format': '',  # No date labels
            'xrotation': 0,
            'tight_layout': True,
            'scale_padding': {'left': 0.05, 'top': 0.05, 'right': 0.05, 'bottom': 0.05},
            'savefig': dict(
                fname=filename,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0,  # No padding for exact sizing
                facecolor='white',
                edgecolor='none'
            )
        }
        
        # Create and save the chart
        fig, axes = mpf.plot(chart_data, **plot_kwargs)
        
        # Remove all text and axes for pure visual input
        if hasattr(axes, '__iter__'):
            ax = axes[0]
        else:
            ax = axes
            
        # Clean up the chart for pure visual learning
        ax.set_xticks([])  # Remove X-axis ticks
        ax.set_yticks([])  # Remove Y-axis ticks
        ax.set_xlabel('')  # Remove X-axis label
        ax.set_ylabel('')  # Remove Y-axis label
        ax.spines['top'].set_visible(False)     # Remove top border
        ax.spines['right'].set_visible(False)   # Remove right border
        ax.spines['bottom'].set_visible(False)  # Remove bottom border
        ax.spines['left'].set_visible(False)    # Remove left border
        
        # Save the figure
        fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='white',
            edgecolor='none'
        )
        
        # Close figure to free memory
        plt.close(fig)
        
        self.logger.debug(f"Pure visual chart saved: {filename} ({self.image_size}x{self.image_size})")
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
    
    def batch_generate_charts(self, data: pd.DataFrame, instrument: str = "NQ", 
                             start_date: str = None, end_date: str = None) -> list:
        """
        Generate pure visual charts for multiple dates in batch.
        
        Args:
            data: Full daily price data
            instrument: Instrument identifier (NQ, ES, YM)
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
        
        self.logger.info(f"Generating pure visual charts for {instrument}: {len(analysis_dates)} trading days")
        
        for i, analysis_date in enumerate(analysis_dates):
            try:
                chart_path = self.generate_chart_image(analysis_date, data, instrument)
                chart_paths.append(chart_path)
                
                if (i + 1) % 100 == 0:  # Log every 100 charts
                    self.logger.info(f"Generated {i + 1}/{len(analysis_dates)} charts for {instrument}")
                    
            except Exception as e:
                error_msg = f"Error generating chart for {instrument} {analysis_date}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        self.logger.info(f"Batch generation complete for {instrument}: {len(chart_paths)} charts, {len(errors)} errors")
        
        if errors:
            self.logger.warning(f"Errors encountered: {errors[:3]}...")  # Log first 3 errors
        
        return chart_paths

def main():
    """Test the pure visual chart generator."""
    print("Testing Pure Visual Chart Generator...")
    
    try:
        # Initialize generator
        generator = PureVisualChartGenerator()
        
        print(f"🎯 Chart target size: {generator.image_size}x{generator.image_size} pixels")
        print(f"📊 Bars per chart: {generator.bars_per_chart}")
        
        # For testing, we'd need actual data - this is just a demo
        print("✅ Pure visual chart generator initialized successfully!")
        print("💡 To test fully, provide daily data for NQ, ES, or YM")
        print("📋 Features:")
        print("   - Clean 224x224 pixel charts for ViT input")
        print("   - No text overlays or labels")  
        print("   - High contrast candlesticks")
        print("   - Green/red previous day level lines")
        print("   - Multi-instrument support (NQ, ES, YM)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())