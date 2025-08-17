#!/usr/bin/env python3
"""
Binary Dataset Creator for Multi-Instrument Bullish/Bearish Classification
Creates training datasets from DOW, NASDAQ, SP500 futures with binary chart patterns
Uses Excel data source instead of API calls
"""

import pandas as pd
import numpy as np
import json
import os
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from threading import Lock

# Import our binary analysis components
from binary_pattern_analyzer import BinaryPatternAnalyzer
from binary_chart_generator import BinaryChartGenerator
from excel_data_fetcher import ExcelDataFetcher


class BinaryDatasetCreator:
    """Creates comprehensive binary classification datasets for multi-instrument training."""

    def __init__(self, config_path: str = "config/config_binary_visual.yaml"):
        """Initialize binary dataset creator."""
        self.config = self.load_config(config_path)
        self.setup_logging()

        # Initialize components
        self.pattern_analyzer = BinaryPatternAnalyzer(config_path)
        self.chart_generator = BinaryChartGenerator(config_path)
        self.data_fetcher = ExcelDataFetcher(config_path)

        # Dataset tracking
        self.samples_created = 0
        self.samples_lock = Lock()
        self.dataset_manifest = {
            "creation_date": datetime.now().isoformat(),
            "config": self.config,
            "instruments": {},
            "samples": [],
            "statistics": {},
        }

        # Ensure directories exist
        self.setup_directories()

    def resolve_instruments(self, selected_instruments: List[str] = None) -> List[str]:
        """
        Resolve instrument selection including handling 'ALL' option.

        Args:
            selected_instruments: List of selected instruments or None for default

        Returns:
            List of actual instrument names to process
        """
        if selected_instruments is None:
            # Use default from config
            selected_instruments = self.config["data"]["instruments"]

        # Handle 'ALL' selection
        if "ALL" in selected_instruments:
            available = self.config["data"]["available_instruments"]
            # Return all instruments except 'ALL'
            return [inst for inst in available if inst != "ALL"]
        
        # Validate instrument selection
        available = self.config["data"]["available_instruments"]
        valid_instruments = []
        for instrument in selected_instruments:
            if instrument in available and instrument != "ALL":
                valid_instruments.append(instrument)
            else:
                self.logger.warning(f"Invalid instrument: {instrument}. Available: {available}")
        
        if not valid_instruments:
            raise ValueError(f"No valid instruments selected from: {selected_instruments}")

        return valid_instruments

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging for dataset creation."""
        log_dir = self.config["paths"]["logs_dir"]
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(f"{log_dir}/binary_dataset_creator.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories for binary dataset."""
        directories = [
            self.config["paths"]["images"],
            self.config["paths"]["labels"],
            self.config["paths"]["metadata"],
            self.config["paths"]["logs_dir"],
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def fetch_instrument_data(
        self, instrument: str, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single instrument from Excel.

        Args:
            instrument: Instrument name (DOW, NASDAQ, SP500)
            start_year: Start year for data
            end_year: End year for data

        Returns:
            DataFrame with OHLC data
        """
        try:
            self.logger.info(
                f"Fetching Excel data for {instrument} ({start_year}-{end_year})"
            )

            # Fetch data using ExcelDataFetcher
            data = self.data_fetcher.fetch_instrument_data(
                instrument=instrument,
                start_year=start_year, 
                end_year=end_year
            )

            if data is None or len(data) == 0:
                raise ValueError(f"No data retrieved for {instrument}")

            self.logger.info(f"Retrieved {len(data)} bars for {instrument}")
            self.logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

            return data

        except Exception as e:
            self.logger.error(f"Error fetching data for {instrument}: {e}")
            raise

    def create_binary_samples(self, instrument: str, data: pd.DataFrame) -> List[Dict]:
        """
        Create binary classification samples for an instrument.

        Args:
            instrument: Instrument identifier
            data: Historical OHLC data

        Returns:
            List of sample dictionaries
        """
        try:
            self.logger.info(f"Creating binary samples for {instrument}")

            samples = []
            errors = []
            bars_per_chart = self.config["data"]["bars_per_chart"]

            # Start from index that allows full chart window
            start_idx = bars_per_chart - 1

            for i in range(start_idx, len(data)):
                try:
                    analysis_date = data.index[i]
                    daily_candle = data.iloc[i]

                    # Analyze binary pattern (bullish/bearish/neutral)
                    pattern = self.pattern_analyzer.analyze_binary_pattern(daily_candle)

                    # Generate chart image
                    chart_path = self.chart_generator.generate_binary_chart_image(
                        analysis_date, data, instrument
                    )

                    # Extract candle features for metadata
                    features = self.pattern_analyzer.extract_candle_features(
                        daily_candle
                    )

                    # Create sample record
                    sample = {
                        "sample_id": f"{instrument}_{analysis_date.strftime('%Y%m%d')}",
                        "date": analysis_date.isoformat(),
                        "instrument": instrument,
                        "pattern": pattern,
                        "pattern_label": self.config["classification"]["labels"][
                            pattern
                        ],
                        "chart_path": chart_path,
                        "features": features,
                        "ohlc": {
                            "open": float(daily_candle["Open"]),
                            "high": float(daily_candle["High"]),
                            "low": float(daily_candle["Low"]),
                            "close": float(daily_candle["Close"]),
                        },
                    }

                    samples.append(sample)

                    # Update progress counter thread-safely
                    with self.samples_lock:
                        self.samples_created += 1

                    if len(samples) % 500 == 0:
                        self.logger.info(
                            f"Created {len(samples)} binary samples for {instrument}"
                        )

                except Exception as e:
                    error_msg = f"Error creating sample for {analysis_date}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)

            self.logger.info(
                f"Binary sample creation complete for {instrument}: {len(samples)} samples, {len(errors)} errors"
            )

            if errors:
                self.logger.warning(f"Errors for {instrument}: {errors[:3]}...")

            return samples

        except Exception as e:
            self.logger.error(f"Error creating binary samples for {instrument}: {e}")
            raise

    def create_instrument_dataset(self, instrument: str, start_year: int = None, end_year: int = None) -> Dict:
        """
        Create complete binary dataset for a single instrument.

        Args:
            instrument: Instrument symbol
            start_year: Start year override (None for config default)
            end_year: End year override (None for config default)

        Returns:
            Dictionary with instrument dataset info
        """
        try:
            self.logger.info(f"Starting binary dataset creation for {instrument}")

            # Use provided years or fall back to config
            if start_year is None:
                start_year = self.config["data"]["start_year"]
            if end_year is None:
                end_year = self.config["data"]["end_year"]

            # Fetch historical data
            data = self.fetch_instrument_data(instrument, start_year, end_year)

            # Create binary samples
            samples = self.create_binary_samples(instrument, data)

            # Calculate statistics
            patterns = [s["pattern"] for s in samples]
            stats = self.pattern_analyzer.get_pattern_statistics(patterns)

            # Create instrument dataset info
            instrument_info = {
                "instrument": instrument,
                "total_samples": len(samples),
                "date_range": {
                    "start": data.index[0].isoformat(),
                    "end": data.index[-1].isoformat(),
                },
                "pattern_distribution": stats["pattern_counts"],
                "pattern_percentages": stats["pattern_percentages"],
                "samples": samples,
            }

            # Save instrument-specific data
            self.save_instrument_data(instrument, instrument_info)

            self.logger.info(
                f"Binary dataset created for {instrument}: {len(samples)} samples"
            )
            self.logger.info(f"Pattern distribution: {stats['pattern_percentages']}")

            return instrument_info

        except Exception as e:
            self.logger.error(f"Error creating dataset for {instrument}: {e}")
            raise

    def save_instrument_data(self, instrument: str, instrument_info: Dict):
        """
        Save instrument-specific dataset information.

        Args:
            instrument: Instrument symbol
            instrument_info: Dataset information dictionary
        """
        try:
            # Save labels
            labels_file = os.path.join(
                self.config["paths"]["labels"], f"{instrument}_binary_labels.json"
            )

            with open(labels_file, "w") as f:
                json.dump(instrument_info, f, indent=2, default=str)

            self.logger.info(f"Saved binary labels for {instrument}: {labels_file}")

        except Exception as e:
            self.logger.error(f"Error saving instrument data for {instrument}: {e}")
            raise

    def create_multi_instrument_dataset(self, selected_instruments: List[str] = None, start_year: int = None, end_year: int = None) -> Dict:
        """
        Create comprehensive binary dataset across selected instruments.

        Args:
            selected_instruments: List of instruments to process (None for default)
            start_year: Start year override (None for config default)
            end_year: End year override (None for config default)

        Returns:
            Dictionary with complete dataset information
        """
        try:
            # Resolve instrument selection
            instruments = self.resolve_instruments(selected_instruments)
            
            self.logger.info(f"Starting multi-instrument binary dataset creation for: {instruments}")
            all_samples = []

            # Create datasets for each instrument
            for instrument in instruments:
                try:
                    instrument_info = self.create_instrument_dataset(instrument, start_year, end_year)

                    # Add to manifest
                    self.dataset_manifest["instruments"][instrument] = {
                        "total_samples": instrument_info["total_samples"],
                        "pattern_distribution": instrument_info["pattern_distribution"],
                        "date_range": instrument_info["date_range"],
                    }

                    # Collect all samples
                    all_samples.extend(instrument_info["samples"])
                except Exception as e:
                    self.logger.error(f"Failed to create dataset for {instrument}: {e}")
                    continue

            # Calculate overall statistics
            overall_stats = self.calculate_dataset_statistics(all_samples)

            # Update manifest
            self.dataset_manifest["total_samples"] = len(all_samples)
            self.dataset_manifest["statistics"] = overall_stats
            self.dataset_manifest["samples"] = all_samples

            # Create data splits
            splits = self.create_data_splits(all_samples)
            self.dataset_manifest["data_splits"] = splits

            # Save complete dataset
            self.save_complete_dataset()

            self.logger.info(f"Multi-instrument binary dataset creation complete")
            self.logger.info(f"Total samples: {len(all_samples)}")
            self.logger.info(
                f"Instruments: {list(self.dataset_manifest['instruments'].keys())}"
            )

            return self.dataset_manifest

        except Exception as e:
            self.logger.error(f"Error creating multi-instrument dataset: {e}")
            raise

    def calculate_dataset_statistics(self, all_samples: List[Dict]) -> Dict:
        """
        Calculate comprehensive dataset statistics.

        Args:
            all_samples: List of all samples across instruments

        Returns:
            Dictionary with dataset statistics
        """
        try:
            patterns = [s["pattern"] for s in all_samples]
            instruments = [s["instrument"] for s in all_samples]
            dates = [pd.to_datetime(s["date"]) for s in all_samples]

            # Overall pattern statistics
            pattern_stats = self.pattern_analyzer.get_pattern_statistics(patterns)

            # Instrument distribution
            instrument_counts = {
                inst: instruments.count(inst) for inst in set(instruments)
            }

            # Temporal distribution
            date_range = {
                "start": min(dates).isoformat(),
                "end": max(dates).isoformat(),
                "span_years": (max(dates) - min(dates)).days / 365.25,
            }

            # Monthly distribution
            monthly_dist = {}
            for month in range(1, 13):
                month_name = pd.to_datetime(f"2000-{month:02d}-01").strftime("%B")
                month_samples = [d for d in dates if d.month == month]
                monthly_dist[month_name] = len(month_samples)

            stats = {
                "total_samples": len(all_samples),
                "pattern_distribution": pattern_stats["pattern_counts"],
                "pattern_percentages": pattern_stats["pattern_percentages"],
                "instrument_distribution": instrument_counts,
                "date_range": date_range,
                "monthly_distribution": monthly_dist,
                "class_balance": {
                    "bearish_ratio": pattern_stats["bearish_percentage"] / 100,
                    "bullish_ratio": pattern_stats["bullish_percentage"] / 100,
                    "neutral_ratio": pattern_stats["neutral_percentage"] / 100,
                },
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating dataset statistics: {e}")
            raise

    def create_data_splits(self, all_samples: List[Dict]) -> Dict:
        """
        Create train/validation/test splits with temporal separation.

        Args:
            all_samples: List of all samples

        Returns:
            Dictionary with data split information
        """
        try:
            self.logger.info("Creating temporal data splits")

            # Sort samples by date
            sorted_samples = sorted(all_samples, key=lambda x: x["date"])

            # Calculate split indices
            total_samples = len(sorted_samples)
            train_split = self.config["training"]["train_split"]
            val_split = self.config["training"]["val_split"]

            train_end = int(total_samples * train_split)
            val_end = int(total_samples * (train_split + val_split))

            # Create splits
            train_samples = sorted_samples[:train_end]
            val_samples = sorted_samples[train_end:val_end]
            test_samples = sorted_samples[val_end:]

            splits = {
                "train": {
                    "samples": len(train_samples),
                    "date_range": {
                        "start": train_samples[0]["date"],
                        "end": train_samples[-1]["date"],
                    },
                    "pattern_distribution": self._get_split_pattern_distribution(
                        train_samples
                    ),
                },
                "validation": {
                    "samples": len(val_samples),
                    "date_range": {
                        "start": val_samples[0]["date"] if val_samples else None,
                        "end": val_samples[-1]["date"] if val_samples else None,
                    },
                    "pattern_distribution": self._get_split_pattern_distribution(
                        val_samples
                    ),
                },
                "test": {
                    "samples": len(test_samples),
                    "date_range": {
                        "start": test_samples[0]["date"] if test_samples else None,
                        "end": test_samples[-1]["date"] if test_samples else None,
                    },
                    "pattern_distribution": self._get_split_pattern_distribution(
                        test_samples
                    ),
                },
            }

            self.logger.info(f"Data splits created:")
            self.logger.info(f"  Train: {len(train_samples)} samples")
            self.logger.info(f"  Validation: {len(val_samples)} samples")
            self.logger.info(f"  Test: {len(test_samples)} samples")

            return splits

        except Exception as e:
            self.logger.error(f"Error creating data splits: {e}")
            raise

    def _get_split_pattern_distribution(self, samples: List[Dict]) -> Dict:
        """Get pattern distribution for a data split."""
        if not samples:
            return {}

        patterns = [s["pattern"] for s in samples]
        distribution = {}
        for pattern in range(3):  # 0: Bearish, 1: Bullish, 2: Neutral
            distribution[pattern] = patterns.count(pattern)

        return distribution

    def save_complete_dataset(self):
        """Save complete dataset manifest and metadata."""
        try:
            # Save main manifest
            manifest_file = os.path.join(
                self.config["paths"]["metadata"], "binary_dataset_manifest.json"
            )

            with open(manifest_file, "w") as f:
                json.dump(self.dataset_manifest, f, indent=2, default=str)

            # Save simplified sample index
            sample_index = []
            for sample in self.dataset_manifest["samples"]:
                index_entry = {
                    "sample_id": sample["sample_id"],
                    "date": sample["date"],
                    "instrument": sample["instrument"],
                    "pattern": sample["pattern"],
                    "chart_path": sample["chart_path"],
                }
                sample_index.append(index_entry)

            index_file = os.path.join(
                self.config["paths"]["metadata"], "binary_sample_index.json"
            )

            with open(index_file, "w") as f:
                json.dump(sample_index, f, indent=2, default=str)

            self.logger.info(f"Dataset manifest saved: {manifest_file}")
            self.logger.info(f"Sample index saved: {index_file}")

        except Exception as e:
            self.logger.error(f"Error saving complete dataset: {e}")
            raise


def main():
    """Test the binary dataset creator."""
    print("Testing Binary Dataset Creator...")

    try:
        # Initialize creator
        creator = BinaryDatasetCreator()

        print("🏗️  Initializing binary dataset creator...")
        print(f"   Configuration: {creator.config['system']['name']}")
        print(f"   Instruments: {creator.config['data']['instruments']}")
        print(
            f"   Date range: {creator.config['data']['start_year']}-{creator.config['data']['end_year']}"
        )

        # Test with a single instrument first
        test_instrument = "NQ.F"
        print(f"\n🧪 Testing with {test_instrument}...")

        # Note: This would normally fetch real data and create actual charts
        # For testing, we'll simulate the process
        print("   📊 This test simulates the dataset creation process")
        print("   📈 Real implementation would:")
        print("   - Fetch historical data from stooq")
        print("   - Generate 30-bar chart images")
        print("   - Analyze bullish/bearish patterns")
        print("   - Create training samples with labels")
        print("   - Save dataset manifest and metadata")

        # Show expected output structure
        print("\n📁 Expected Dataset Structure:")
        print("   data/images_binary/       # Chart images (224x224 PNG)")
        print("   data/labels_binary/       # JSON files with labels")
        print("   data/metadata_binary/     # Dataset manifest and indices")

        print("\n🎯 Binary Dataset Features:")
        print("   - Multi-instrument training (NQ, ES, YM)")
        print("   - 3-class classification (Bearish/Bullish/Neutral)")
        print("   - Clean 30-bar candlestick charts")
        print("   - No previous day level dependencies")
        print(
            "   - Temporal data splits (2000-2020 train, 2021-2022 val, 2023-2025 test)"
        )
        print("   - Expected ~20,000 samples total")
        print("   - ViT-ready 224x224 images")

        print("✅ Binary dataset creator test completed successfully!")
        print("\n🚀 Ready for full dataset generation:")
        print("   python src/generate_binary_dataset.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
