#!/usr/bin/env python3
"""
Daily Visual Chart Generator for 4-Class Previous Day Levels Analysis
Generates 30-bar daily candlestick charts WITH previous day level reference lines
Optimized for 4-class classification with hybrid ViT models
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


class DailyChartGenerator:
    """Generates 30-bar daily charts with previous day level reference lines for 4-class classification."""

    def __init__(self, config_path: str = "config/config_daily_hybrid.yaml"):
        """Initialize daily chart generator with configuration."""
        self.config = self.load_config(config_path)
        self.chart_config = self.config["chart"]
        self.bars_per_chart = self.config["data"]["bars_per_chart"]
        self.image_size = self.chart_config["image_size"]
        self.setup_logging()

        # Ensure output directory exists
        os.makedirs(self.config["paths"]["images"], exist_ok=True)

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging for the chart generator."""
        log_dir = self.config["paths"]["logs_dir"]
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(f"{log_dir}/daily_chart_generator.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def generate_daily_chart_image(
        self, 
        analysis_date: datetime, 
        data: pd.DataFrame, 
        previous_candle: pd.Series, 
        instrument: str = "NQ"
    ) -> str:
        """
        Generate PREDICTIVE daily chart with previous day level reference lines.
        Shows 30 historical bars EXCLUDING the analysis date for true prediction.

        Args:
            analysis_date: The date being predicted (NOT shown in chart)
            data: Full daily price data
            previous_candle: Previous day OHLC data for reference lines
            instrument: Instrument identifier (NQ, ES, YM)

        Returns:
            Path to saved chart image showing historical context only
        """
        try:
            self.logger.info(
                f"Generating PREDICTIVE chart for {instrument} predicting {analysis_date.date()}"
            )

            # Get 30-bar chart window EXCLUDING analysis date (true prediction)
            chart_data = self.get_chart_window(analysis_date, data)

            # Validate chart data
            if not self.validate_chart_data(chart_data):
                raise ValueError(f"Invalid chart data for {analysis_date}")

            # Extract previous day levels for reference lines
            prev_high = float(previous_candle["High"])
            prev_low = float(previous_candle["Low"])

            # Generate predictive chart with reference lines (224x224 for ViT)
            chart_path = self.create_daily_chart(
                chart_data, analysis_date, instrument, prev_high, prev_low
            )

            self.logger.info(f"Predictive chart generated: {chart_path}")
            return chart_path

        except Exception as e:
            self.logger.error(f"Error generating daily chart for {analysis_date}: {e}")
            raise

    def get_chart_window(
        self, analysis_date: datetime, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get 30-day chart window EXCLUDING the analysis date for true prediction.
        Shows previous 30 days with reference lines, but NOT the current day being predicted.

        Args:
            analysis_date: Date being analyzed (excluded from chart)
            data: Full daily price data

        Returns:
            Chart data for 30 bars BEFORE analysis date
        """
        # Convert to datetime if needed
        if isinstance(analysis_date, str):
            analysis_date = pd.to_datetime(analysis_date)

        # Find position of analysis date
        try:
            analysis_idx = data.index.get_indexer([analysis_date])[0]
        except (KeyError, IndexError):
            raise ValueError(f"Analysis date {analysis_date} not found in data")

        # Calculate start index for 30-bar window BEFORE analysis date
        # We want 30 bars ending the day BEFORE analysis_date
        start_idx = max(0, analysis_idx - self.bars_per_chart)
        end_idx = analysis_idx  # EXCLUDE analysis_date from chart

        # Extract chart window (previous 30 days only)
        chart_data = data.iloc[start_idx:end_idx].copy()

        if len(chart_data) < self.bars_per_chart:
            self.logger.warning(
                f"Chart window only has {len(chart_data)} bars (need {self.bars_per_chart})"
            )

        self.logger.debug(
            f"Predictive chart window: {chart_data.index[0]} to {chart_data.index[-1]} ({len(chart_data)} bars) - EXCLUDES analysis date {analysis_date.date()}"
        )

        return chart_data

    def validate_chart_data(self, chart_data: pd.DataFrame) -> bool:
        """
        Validate chart data quality for 4-class classification.

        Args:
            chart_data: Chart data to validate

        Returns:
            True if data is valid
        """
        if chart_data is None or len(chart_data) == 0:
            self.logger.error("Chart data is empty")
            return False

        # Check for required columns (including volume for enhanced charts)
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in chart_data.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False

        # Check for NaN values
        if chart_data[required_columns].isnull().any().any():
            self.logger.error("Chart data contains NaN values")
            return False

        # Check for valid price relationships
        for idx, row in chart_data.iterrows():
            if not (
                row["Low"] <= row["Open"] <= row["High"]
                and row["Low"] <= row["Close"] <= row["High"]
            ):
                self.logger.error(f"Invalid OHLC relationship on {idx}")
                return False

        # Check minimum bars
        if len(chart_data) < 10:  # Minimum bars for meaningful chart
            self.logger.error(f"Not enough bars for chart: {len(chart_data)}")
            return False

        return True

    def create_daily_chart(
        self, 
        chart_data: pd.DataFrame, 
        analysis_date: datetime, 
        instrument: str,
        prev_high: float,
        prev_low: float
    ) -> str:
        """
        Create PREDICTIVE daily chart with previous day level reference lines.
        Shows historical context only - the analysis date candle is NOT included.

        Args:
            chart_data: 30-bar historical data (EXCLUDES analysis date)
            analysis_date: Date being predicted (for filename only)
            instrument: Instrument identifier
            prev_high: Previous day high level (green line)
            prev_low: Previous day low level (red line)

        Returns:
            Path to saved predictive chart image
        """
        # Configure mplfinance style for 4-class classification
        style_config = {
            "base_mpl_style": "seaborn-v0_8",
            "marketcolors": mpf.make_marketcolors(
                up="#2ca02c",  # Green for bullish candles
                down="#d62728",  # Red for bearish candles
                edge="inherit",
                wick={"up": "#2ca02c", "down": "#d62728"},
                volume="in",
            ),
            "gridcolor": "#f0f0f0",
            "gridstyle": "-",
            "y_on_right": False,
            "facecolor": "white",
        }

        # Create custom style
        daily_style = mpf.make_mpf_style(**style_config)

        # Create horizontal lines for previous day levels
        hlines = dict(
            hlines=[prev_high, prev_low],
            colors=["#2ca02c", "#d62728"],  # Green for prev high, red for prev low
            linestyle=["-", "-"],
            linewidths=[1.5, 1.5],
            alpha=[0.8, 0.8]
        )

        # Enhanced plot settings optimized for higher resolution (448x448 support)
        volume_enabled = self.chart_config.get("volume", True)
        
        # Adjust line widths and candle widths based on image resolution
        line_scale = self.image_size / 224.0  # Scale factor (1.0 for 224, 2.0 for 448)
        candle_linewidth = max(0.8, 1.0 * line_scale)
        candle_width = min(0.8, max(0.4, 0.6 * line_scale))
        
        plot_config = {
            "type": "candle",
            "style": daily_style,
            "figsize": (self.chart_config["width"], self.chart_config["height"]),
            "volume": volume_enabled,
            "ylabel": "",
            "ylabel_lower": "",
            "tight_layout": True,
            "scale_padding": {"left": 0.3, "top": 0.8, "right": 0.5, "bottom": 0.8},
            "panel_ratios": (3, 1) if volume_enabled else (1,),  # Conditional volume panel
            "figratio": (1, 1),  # Square aspect ratio for ViT input
            "figscale": max(1.0, line_scale * 0.8),  # Scale for higher resolution
            "update_width_config": dict(
                candle_linewidth=candle_linewidth, 
                candle_width=candle_width
            ),
            "hlines": hlines,  # Add previous day level reference lines
        }

        # Add title only if enabled
        if self.chart_config.get("title", False):
            plot_config["title"] = f"{instrument} Daily Pattern"

        # Generate filename
        date_str = analysis_date.strftime("%Y%m%d")
        filename = f"daily_chart_{instrument}_{date_str}.png"
        chart_path = os.path.join(self.config["paths"]["images"], filename)

        try:
            # Create the daily chart with reference lines
            fig, axes = mpf.plot(
                chart_data,
                returnfig=True,
                savefig=dict(
                    fname=chart_path,
                    dpi=self.chart_config["dpi"],
                    bbox_inches="tight",
                    pad_inches=0.1,
                    facecolor="white",
                    edgecolor="none",
                ),
                **plot_config,
            )

            # Add reference line labels if enabled (scaled for resolution)
            if self.chart_config.get("reference_labels", True):
                # Scale font size based on image resolution
                label_fontsize = max(8, int(8 * line_scale))
                
                for ax in axes:
                    # Add text labels for previous day levels
                    y_range = ax.get_ylim()
                    x_range = ax.get_xlim()
                    
                    # Position labels on the right side
                    label_x = x_range[1] - (x_range[1] - x_range[0]) * 0.05
                    
                    # Previous high label (green) - enhanced for higher resolution
                    if y_range[0] <= prev_high <= y_range[1]:
                        ax.text(label_x, prev_high, "Prev H", 
                               color="#2ca02c", fontsize=label_fontsize, fontweight="bold",
                               ha="right", va="center", 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                       alpha=0.9, edgecolor="#2ca02c", linewidth=1))
                    
                    # Previous low label (red) - enhanced for higher resolution
                    if y_range[0] <= prev_low <= y_range[1]:
                        ax.text(label_x, prev_low, "Prev L", 
                               color="#d62728", fontsize=label_fontsize, fontweight="bold",
                               ha="right", va="center",
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                       alpha=0.9, edgecolor="#d62728", linewidth=1))

            # Clean up the chart for 4-class classification
            if not self.chart_config.get("axes_labels", True):
                # Remove axis labels for cleaner appearance
                for ax in axes:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.tick_params(labelbottom=False, labelleft=False)

                    # Remove grid if specified
                    if not self.chart_config.get("grid", True):
                        ax.grid(False)

            plt.close(fig)

            # Resize to exact ViT input size (224x224)
            self.resize_chart_image(chart_path)

            self.logger.debug(f"Daily chart with reference lines saved: {chart_path}")
            self.logger.debug(f"Previous day levels - High: {prev_high:.2f}, Low: {prev_low:.2f}")
            return chart_path

        except Exception as e:
            self.logger.error(f"Error creating daily chart: {e}")
            raise

    def resize_chart_image(self, chart_path: str):
        """
        Resize chart image to exact ViT input size (224x224 or 448x448 for pure visual).

        Args:
            chart_path: Path to chart image
        """
        try:
            from PIL import Image

            # Open and resize image
            with Image.open(chart_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to exact ViT input size (supports both 224 and 448)
                resized_img = img.resize(
                    (self.image_size, self.image_size), Image.Resampling.LANCZOS
                )

                # For higher resolution (448x448), use higher quality
                quality = 98 if self.image_size >= 448 else 95
                
                # Save resized image
                resized_img.save(chart_path, "PNG", quality=quality, optimize=True)

            self.logger.debug(f"Chart resized to {self.image_size}x{self.image_size} (quality={quality})")

        except ImportError:
            self.logger.warning("PIL not available for image resizing")
        except Exception as e:
            self.logger.error(f"Error resizing chart image: {e}")
            raise

    def generate_batch_charts(
        self, 
        analysis_data: list, 
        data: pd.DataFrame, 
        instrument: str = "NQ"
    ) -> list:
        """
        Generate daily charts for multiple dates in batch.

        Args:
            analysis_data: List of tuples (analysis_date, previous_candle)
            data: Full daily price data
            instrument: Instrument identifier

        Returns:
            List of generated chart paths
        """
        chart_paths = []
        errors = []

        self.logger.info(f"Generating {len(analysis_data)} daily charts for {instrument}")

        for i, (analysis_date, previous_candle) in enumerate(analysis_data):
            try:
                chart_path = self.generate_daily_chart_image(
                    analysis_date, data, previous_candle, instrument
                )
                chart_paths.append(chart_path)

                if (i + 1) % 100 == 0:
                    self.logger.info(f"Generated {i + 1}/{len(analysis_data)} daily charts")

            except Exception as e:
                error_msg = f"Error generating chart for {analysis_date}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        self.logger.info(
            f"Batch generation complete: {len(chart_paths)} charts, {len(errors)} errors"
        )

        if errors:
            self.logger.warning(f"Errors encountered: {errors[:3]}...")

        return chart_paths

    def cleanup_old_charts(self, days_old: int = 30):
        """
        Clean up old chart images to save disk space.

        Args:
            days_old: Delete charts older than this many days
        """
        try:
            chart_dir = Path(self.config["paths"]["images"])
            cutoff_date = datetime.now() - timedelta(days=days_old)

            deleted_count = 0
            for chart_file in chart_dir.glob("daily_chart_*.png"):
                if chart_file.stat().st_mtime < cutoff_date.timestamp():
                    chart_file.unlink()
                    deleted_count += 1

            self.logger.info(f"Cleaned up {deleted_count} old daily charts")

        except Exception as e:
            self.logger.error(f"Error cleaning up charts: {e}")


def main():
    """Test the daily chart generator."""
    print("Testing Daily Chart Generator...")

    try:
        # Initialize generator
        generator = DailyChartGenerator()

        # Create test data (simplified OHLC)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        price = 15000
        test_data = []

        for date in dates:
            # Random walk for realistic price movement
            change = np.random.normal(0, 50)
            price += change

            # Generate OHLC
            open_price = price
            high_price = open_price + abs(np.random.normal(0, 30))
            low_price = open_price - abs(np.random.normal(0, 30))
            close_price = open_price + np.random.normal(0, 40)

            # Ensure valid OHLC relationships
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # Generate realistic volume (higher volume on larger price moves)
            price_change = abs(close_price - open_price)
            base_volume = np.random.lognormal(15, 0.5)  # Realistic volume distribution
            volume_multiplier = 1 + (price_change / open_price) * 10  # Higher volume on bigger moves
            volume = int(base_volume * volume_multiplier)

            test_data.append(
                {
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                }
            )

            price = close_price

        test_df = pd.DataFrame(test_data, index=dates)

        print(f"📊 Created test data with {len(test_df)} daily bars")
        print(f"   Date range: {test_df.index[0].date()} to {test_df.index[-1].date()}")

        # Test single chart generation
        test_date = test_df.index[-10]  # 10 days from end
        previous_candle = test_df.iloc[-11]  # Previous day

        print(f"🎨 Generating daily chart for {test_date.date()}...")
        print(f"   Previous day levels - High: {previous_candle['High']:.2f}, Low: {previous_candle['Low']:.2f}")
        
        chart_path = generator.generate_daily_chart_image(test_date, test_df, previous_candle, "NQ")

        print(f"✅ Daily chart generated: {chart_path}")

        # Verify chart file exists and get info
        if os.path.exists(chart_path):
            file_size = os.path.getsize(chart_path) / 1024  # KB
            print(f"   📁 File size: {file_size:.1f} KB")

            # Check image dimensions if PIL available
            try:
                from PIL import Image

                with Image.open(chart_path) as img:
                    print(f"   📐 Image dimensions: {img.size}")
                    print(f"   🎨 Image mode: {img.mode}")
            except ImportError:
                print("   📸 PIL not available for image verification")

        # Test batch generation (last 5 days)
        print(f"\n📈 Testing batch generation...")
        batch_data = []
        for i in range(-5, 0):  # Last 5 days
            analysis_date = test_df.index[i]
            previous_candle = test_df.iloc[i-1]
            batch_data.append((analysis_date, previous_candle))
            
        batch_paths = generator.generate_batch_charts(batch_data, test_df, "NQ")

        print(f"✅ Batch generation complete: {len(batch_paths)} charts")

        print("🎯 Daily Chart Features:")
        print("   - 30-bar candlestick charts with previous day levels")
        print("   - Green horizontal line for previous day high")
        print("   - Red horizontal line for previous day low")
        print("   - 224x224 pixel images for ViT")
        print("   - Optimized for 4-class pattern classification")
        print("   - Green/red candle color coding")
        print("   - Reference line labels for clarity")

        print("✅ Daily chart generator test completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
