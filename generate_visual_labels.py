#!/usr/bin/env python3
"""
Generate labels for existing daily chart images
Creates the pure visual dataset structure needed for training
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import yaml
import logging

# Add src to path for imports
sys.path.append("src")

from stooq_fetcher import DailyNQFetcher
from daily_pattern_analyzer import DailyPatternAnalyzer


def setup_logging():
    """Setup logging for label generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def extract_date_from_filename(filename):
    """Extract date from image filename like 'daily_chart_20001030.png'"""
    # Remove prefix and suffix
    date_str = filename.replace("daily_chart_", "").replace(".png", "")
    # Convert YYYYMMDD to YYYY-MM-DD
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def main():
    logger = setup_logging()
    logger.info("Starting visual label generation...")

    # Initialize components
    data_fetcher = DailyNQFetcher()
    pattern_analyzer = DailyPatternAnalyzer()

    # Get list of existing images
    images_dir = Path("data/images")
    if not images_dir.exists():
        logger.error("Images directory not found!")
        return 1

    image_files = list(images_dir.glob("daily_chart_*.png"))
    logger.info(f"Found {len(image_files)} image files")

    # Create labels directory
    labels_dir = Path("data/labels")
    labels_dir.mkdir(exist_ok=True)

    # Fetch all market data
    logger.info("Fetching market data...")
    market_data = data_fetcher.fetch_daily_data(start_year=2000, end_year=2025)
    logger.info(f"Market data: {len(market_data)} trading days")

    # Generate labels for each image
    labels_created = 0
    errors = []

    for image_file in image_files:
        try:
            # Extract date from filename
            date_str = extract_date_from_filename(image_file.name)
            analysis_date = pd.to_datetime(date_str)

            # Get market data for this date
            if analysis_date not in market_data.index:
                logger.warning(f"No market data for {date_str}")
                continue

            # Get previous trading day
            current_idx = market_data.index.get_indexer([analysis_date])[0]
            if current_idx <= 0:
                logger.warning(f"No previous day data for {date_str}")
                continue

            # Get current and previous day data
            current_day = market_data.iloc[current_idx]
            prev_day = market_data.iloc[current_idx - 1]

            prev_high = float(prev_day["High"])
            prev_low = float(prev_day["Low"])

            # Analyze pattern
            pattern = pattern_analyzer.analyze_daily_pattern(
                current_day, prev_high, prev_low
            )

            # Get pattern label
            pattern_labels = {
                1: "High Breakout",
                2: "Low Breakdown",
                3: "Range Expansion",
                4: "Range Bound",
            }

            # Create label data
            label_data = {
                "date": date_str,
                "pattern_rank": pattern,
                "pattern_label": pattern_labels[pattern],
                "chart_image": str(image_file),
                "prev_high": prev_high,
                "prev_low": prev_low,
                "daily_ohlc": {
                    "open": float(current_day["Open"]),
                    "high": float(current_day["High"]),
                    "low": float(current_day["Low"]),
                    "close": float(current_day["Close"]),
                },
            }

            # Save label file
            label_filename = f"daily_sample_{date_str.replace('-', '')}.json"
            label_path = labels_dir / label_filename

            with open(label_path, "w") as f:
                json.dump(label_data, f, indent=2)

            labels_created += 1

            if labels_created % 100 == 0:
                logger.info(f"Created {labels_created} labels...")

        except Exception as e:
            error_msg = f"Error processing {image_file.name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    logger.info(f"Label generation complete!")
    logger.info(f"  Created: {labels_created} labels")
    logger.info(f"  Errors: {len(errors)}")

    if errors:
        logger.warning("Sample errors:")
        for error in errors[:3]:
            logger.warning(f"  {error}")

    # Calculate pattern distribution
    pattern_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for label_file in labels_dir.glob("daily_sample_*.json"):
        with open(label_file, "r") as f:
            data = json.load(f)
            pattern_counts[data["pattern_rank"]] += 1

    total_samples = sum(pattern_counts.values())
    logger.info(f"Pattern distribution:")
    for pattern, count in pattern_counts.items():
        label = pattern_labels[pattern]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        logger.info(f"  {pattern}: {label} - {count} ({percentage:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
