#!/usr/bin/env python3
"""
Binary Visual Chart Generator for Bullish/Bearish Pattern Analysis
Generates clean 30-bar daily candlestick charts without previous day levels
Optimized for binary classification with ViT models
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


class BinaryChartGenerator:
    """Generates clean 30-bar daily charts optimized for binary bullish/bearish classification."""

    def __init__(self, config_path: str = "config/config_binary_visual.yaml"):
        """Initialize binary chart generator with configuration."""
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
                logging.FileHandler(f"{log_dir}/binary_chart_generator.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def generate_binary_chart_image(
        self, analysis_date: datetime, data: pd.DataFrame, instrument: str = "NQ"
    ) -> str:
        """
        Generate binary visual 30-bar daily chart for bullish/bearish classification.

        Args:
            analysis_date: The date being analyzed
            data: Full daily price data
            instrument: Instrument identifier (NQ, ES, YM)

        Returns:
            Path to saved chart image
        """
        try:
            self.logger.info(
                f"Generating binary chart for {instrument} on {analysis_date.date()}"
            )

            # Get 30-bar chart window (no previous day levels needed)
            chart_data = self.get_chart_window(analysis_date, data)

            # Validate chart data
            if not self.validate_chart_data(chart_data):
                raise ValueError(f"Invalid chart data for {analysis_date}")

            # Generate clean binary chart (224x224 for ViT)
            chart_path = self.create_binary_chart(chart_data, analysis_date, instrument)

            self.logger.info(f"Binary chart generated: {chart_path}")
            return chart_path

        except Exception as e:
            self.logger.error(f"Error generating binary chart for {analysis_date}: {e}")
            raise

    def get_chart_window(
        self, analysis_date: datetime, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get 30-day chart window for binary classification.

        Args:
            analysis_date: Date being analyzed
            data: Full daily price data

        Returns:
            Chart data for 30 bars
        """
        # Convert to datetime if needed
        if isinstance(analysis_date, str):
            analysis_date = pd.to_datetime(analysis_date)

        # Find position of analysis date
        try:
            analysis_idx = data.index.get_indexer([analysis_date])[0]
        except (KeyError, IndexError):
            raise ValueError(f"Analysis date {analysis_date} not found in data")

        # Calculate start index for 30-bar window
        start_idx = max(0, analysis_idx - self.bars_per_chart + 1)
        end_idx = analysis_idx + 1

        # Extract chart window
        chart_data = data.iloc[start_idx:end_idx].copy()

        if len(chart_data) < self.bars_per_chart:
            self.logger.warning(
                f"Chart window only has {len(chart_data)} bars (need {self.bars_per_chart})"
            )

        self.logger.debug(
            f"Chart window: {chart_data.index[0]} to {chart_data.index[-1]} ({len(chart_data)} bars)"
        )

        return chart_data

    def validate_chart_data(self, chart_data: pd.DataFrame) -> bool:
        """
        Validate chart data quality for binary classification.

        Args:
            chart_data: Chart data to validate

        Returns:
            True if data is valid
        """
        if chart_data is None or len(chart_data) == 0:
            self.logger.error("Chart data is empty")
            return False

        # Check for required columns
        required_columns = ["Open", "High", "Low", "Close"]
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

    def create_binary_chart(
        self, chart_data: pd.DataFrame, analysis_date: datetime, instrument: str
    ) -> str:
        """
        Create clean binary chart without previous day levels.

        Args:
            chart_data: 30-bar chart data
            analysis_date: Date being analyzed
            instrument: Instrument identifier

        Returns:
            Path to saved chart image
        """
        # Configure mplfinance style for binary classification
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
        binary_style = mpf.make_mpf_style(**style_config)

        # Configure plot settings for binary classification
        plot_config = {
            "type": "candle",
            "style": binary_style,
            "figsize": (self.chart_config["width"], self.chart_config["height"]),
            "volume": self.chart_config["volume"],
            "ylabel": "",
            "ylabel_lower": "",
            "tight_layout": True,
            "scale_padding": {"left": 0.3, "top": 0.8, "right": 0.5, "bottom": 0.8},
            "panel_ratios": (1,),
            "figratio": (1, 1),  # Square aspect ratio for 224x224
            "figscale": 1.0,
            "update_width_config": dict(candle_linewidth=1.0, candle_width=0.6),
        }

        # Add title only if enabled
        if self.chart_config.get("title", False):
            plot_config["title"] = f"{instrument} Binary Pattern"

        # Generate filename
        date_str = analysis_date.strftime("%Y%m%d")
        filename = f"binary_chart_{instrument}_{date_str}.png"
        chart_path = os.path.join(self.config["paths"]["images"], filename)

        try:
            # Create the binary chart
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

            # Clean up the chart for binary classification
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

            self.logger.debug(f"Binary chart saved: {chart_path}")
            return chart_path

        except Exception as e:
            self.logger.error(f"Error creating binary chart: {e}")
            raise

    def resize_chart_image(self, chart_path: str):
        """
        Resize chart image to exact ViT input size (224x224).

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

                # Resize to exact ViT input size
                resized_img = img.resize(
                    (self.image_size, self.image_size), Image.Resampling.LANCZOS
                )

                # Save resized image
                resized_img.save(chart_path, "PNG", quality=95)

            self.logger.debug(f"Chart resized to {self.image_size}x{self.image_size}")

        except ImportError:
            self.logger.warning("PIL not available for image resizing")
        except Exception as e:
            self.logger.error(f"Error resizing chart image: {e}")
            raise

    def generate_batch_charts(
        self, dates: list, data: pd.DataFrame, instrument: str = "NQ"
    ) -> list:
        """
        Generate binary charts for multiple dates in batch.

        Args:
            dates: List of analysis dates
            data: Full daily price data
            instrument: Instrument identifier

        Returns:
            List of generated chart paths
        """
        chart_paths = []
        errors = []

        self.logger.info(f"Generating {len(dates)} binary charts for {instrument}")

        for i, date in enumerate(dates):
            try:
                chart_path = self.generate_binary_chart_image(date, data, instrument)
                chart_paths.append(chart_path)

                if (i + 1) % 100 == 0:
                    self.logger.info(f"Generated {i + 1}/{len(dates)} binary charts")

            except Exception as e:
                error_msg = f"Error generating chart for {date}: {e}"
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
            for chart_file in chart_dir.glob("binary_chart_*.png"):
                if chart_file.stat().st_mtime < cutoff_date.timestamp():
                    chart_file.unlink()
                    deleted_count += 1

            self.logger.info(f"Cleaned up {deleted_count} old binary charts")

        except Exception as e:
            self.logger.error(f"Error cleaning up charts: {e}")


def main():
    """Test the binary chart generator."""
    print("Testing Binary Chart Generator...")

    try:
        # Initialize generator
        generator = BinaryChartGenerator()

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

            test_data.append(
                {
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                }
            )

            price = close_price

        test_df = pd.DataFrame(test_data, index=dates)

        print(f"📊 Created test data with {len(test_df)} daily bars")
        print(f"   Date range: {test_df.index[0].date()} to {test_df.index[-1].date()}")

        # Test single chart generation
        test_date = test_df.index[-10]  # 10 days from end

        print(f"🎨 Generating binary chart for {test_date.date()}...")
        chart_path = generator.generate_binary_chart_image(test_date, test_df, "NQ")

        print(f"✅ Binary chart generated: {chart_path}")

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
        batch_dates = test_df.index[-5:].tolist()
        batch_paths = generator.generate_batch_charts(batch_dates, test_df, "NQ")

        print(f"✅ Batch generation complete: {len(batch_paths)} charts")

        print("🎯 Binary Chart Features:")
        print("   - Clean 30-bar candlestick charts")
        print("   - No previous day level lines")
        print("   - 224x224 pixel images for ViT")
        print("   - Optimized for bullish/bearish classification")
        print("   - Green/red candle color coding")
        print("   - Minimal visual noise")

        print("✅ Binary chart generator test completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
