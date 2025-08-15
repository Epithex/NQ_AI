# src/chart_generator.py
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import yaml
import logging
import os
from typing import Optional

class ChartGenerator:
    """Generates candlestick charts with previous day levels."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ChartGenerator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.chart_config = self.config['chart']
        self.paths = self.config['paths']
        
        # Setup logging
        logging.basicConfig(
            filename=f"{self.config['paths']['logs']}/chart_generator.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Ensure image directory exists
        os.makedirs(self.paths['images'], exist_ok=True)
    
    def generate_chart_image(self, data: pd.DataFrame, prev_high: float, prev_low: float, 
                           timestamp: datetime) -> str:
        """
        Create chart with color-coded previous day levels.
        
        Args:
            data: 300 bars of OHLCV data
            prev_high: Previous day high (green line)
            prev_low: Previous day low (red line)
            timestamp: Analysis timestamp for filename
            
        Returns:
            Path to saved chart image
        """
        try:
            self.logger.info(f"Generating chart for {timestamp}")
            
            # Prepare horizontal lines for previous day levels
            hlines = dict(
                hlines=[prev_high, prev_low],
                colors=[self.chart_config['prev_high_color'], 
                       self.chart_config['prev_low_color']],
                linestyle='-',
                linewidths=self.chart_config['line_width']
            )
            
            # Configure chart style
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='inherit',
                wick={'up':'green', 'down':'red'},
                volume='inherit'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='lightgray'
            )
            
            # Generate filename
            date_str = timestamp.strftime('%Y%m%d_%H%M')
            filename = f"{self.paths['images']}/chart_{date_str}.png"
            
            # Create the chart
            fig, axes = mpf.plot(
                data,
                type='candle',
                style=style,
                hlines=hlines,
                volume=False,
                figsize=(self.chart_config['width'], self.chart_config['height']),
                returnfig=True,
                datetime_format='%m/%d %H:%M',
                xrotation=45,
                tight_layout=True
            )
            
            # Add title with levels info
            axes[0].set_title(
                f"NQ Futures - {timestamp.strftime('%Y-%m-%d %H:%M')} EST\n"
                f"Prev High (Green): {prev_high:.2f} | Prev Low (Red): {prev_low:.2f}",
                fontsize=12,
                pad=20
            )
            
            # Save the chart
            fig.savefig(filename, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Chart saved: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error generating chart: {str(e)}")
            raise
    
    def validate_chart_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data is suitable for chart generation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid for charting
        """
        # Check minimum bars requirement
        if len(data) < 50:  # Minimum bars for meaningful chart
            self.logger.warning(f"Insufficient data: {len(data)} bars")
            return False
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            self.logger.warning("Missing required OHLC columns")
            return False
        
        # Check index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Index is not DatetimeIndex")
            return False
        
        return True
    
    def add_annotations(self, fig, axes, prev_high: float, prev_low: float):
        """
        Add text annotations for levels on the chart.
        
        Args:
            fig: Matplotlib figure
            axes: Chart axes
            prev_high: Previous day high
            prev_low: Previous day low
        """
        # Add text labels for the levels
        y_range = axes[0].get_ylim()
        x_pos = len(axes[0].lines[0].get_xdata()) * 0.98
        
        # High label
        axes[0].text(x_pos, prev_high, f'PDH: {prev_high:.2f}',
                    color='green', fontweight='bold',
                    va='bottom', ha='right')
        
        # Low label
        axes[0].text(x_pos, prev_low, f'PDL: {prev_low:.2f}',
                    color='red', fontweight='bold',
                    va='top', ha='right')