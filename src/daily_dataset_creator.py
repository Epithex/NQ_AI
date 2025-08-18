#!/usr/bin/env python3
"""
Daily Dataset Creator for Multi-Instrument 4-Class Previous Day Levels Classification
Creates hybrid training datasets (visual + numerical) from DOW, NASDAQ, SP500 futures
Uses Excel data source with 4-class pattern analysis and previous day level reference lines
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
import multiprocessing as mp
from multiprocessing import Value, Array
import time

# Import our daily analysis components
from daily_pattern_analyzer import DailyPatternAnalyzer
from daily_chart_generator import DailyChartGenerator
from excel_data_fetcher import ExcelDataFetcher


def process_sample_parallel(args: Tuple) -> Dict:
    """
    Process a single sample in parallel worker process for 4-class hybrid dataset.
    
    Args:
        args: Tuple containing (date, instrument, current_data, previous_data, config_path)
        
    Returns:
        Dictionary with sample information including visual and numerical features
    """
    try:
        date, instrument, current_data, previous_data, config_path = args
        
        # Initialize components in worker process (can't share across processes)
        pattern_analyzer = DailyPatternAnalyzer(config_path)
        chart_generator = DailyChartGenerator(config_path)
        
        # Get current and previous day candles
        current_candle = current_data.iloc[-1]  # Last row is the analysis date
        previous_candle = previous_data.iloc[-1]  # Previous day data
        
        # Analyze 4-class daily pattern
        pattern = pattern_analyzer.analyze_daily_pattern(current_candle, previous_candle)
        
        # Generate chart image with previous day level reference lines
        chart_path = chart_generator.generate_daily_chart_image(
            date, current_data, previous_candle, instrument
        )
        
        # Extract comprehensive features for metadata
        features = pattern_analyzer.extract_candle_features(current_candle, previous_candle)
        
        # Extract numerical features for the hybrid model
        numerical_features = pattern_analyzer.extract_numerical_features(current_candle, previous_candle)
        
        # Create hybrid sample record (visual + numerical)
        sample = {
            "sample_id": f"{instrument}_{date.strftime('%Y%m%d')}",
            "date": date.isoformat(),
            "instrument": instrument,
            "pattern": pattern,
            "pattern_label": {1: "High Breakout", 2: "Low Breakdown", 3: "Range Expansion", 4: "Range Bound"}[pattern],
            "chart_path": chart_path,
            "features": features,
            "numerical_features": numerical_features,  # The 3 key features for hybrid model
            "current_ohlc": {
                "open": float(current_candle["Open"]),
                "high": float(current_candle["High"]),
                "low": float(current_candle["Low"]),
                "close": float(current_candle["Close"]),
            },
            "previous_levels": {
                "high": float(previous_candle["High"]),
                "low": float(previous_candle["Low"]),
            },
            "success": True
        }
        
        return sample
        
    except Exception as e:
        return {
            "date": date.isoformat() if 'date' in locals() else "unknown",
            "instrument": instrument if 'instrument' in locals() else "unknown",
            "error": str(e),
            "success": False
        }


class DailyDatasetCreator:
    """Creates comprehensive 4-class hybrid datasets for multi-instrument training."""

    def __init__(self, config_path: str = "config/config_daily_hybrid.yaml", workers: int = None):
        """Initialize daily dataset creator."""
        self.config = self.load_config(config_path)
        self.config_path = config_path
        self.workers = workers
        self.setup_logging()

        # Initialize components
        self.pattern_analyzer = DailyPatternAnalyzer(config_path)
        self.chart_generator = DailyChartGenerator(config_path)
        self.data_fetcher = ExcelDataFetcher(config_path)

        # Dataset tracking
        self.samples_created = 0
        self.samples_lock = Lock()
        
        # Parallel processing setup
        if self.workers and self.workers > 1:
            self.logger.info(f"Parallel processing enabled: {self.workers} workers")
        else:
            self.logger.info("Single-threaded processing enabled")
        self.dataset_manifest = {
            "creation_date": datetime.now().isoformat(),
            "config": self.config,
            "instruments": {},
            "samples": [],
            "statistics": {},
            "model_type": "hybrid_4class",
            "features": {
                "visual": "30-bar daily charts with previous day level reference lines",
                "numerical": ["distance_to_prev_high", "distance_to_prev_low", "prev_day_range"]
            }
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
                logging.FileHandler(f"{log_dir}/daily_dataset_creator.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories for daily hybrid dataset."""
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

    def create_daily_samples(self, instrument: str, data: pd.DataFrame) -> List[Dict]:
        """
        Create 4-class hybrid classification samples for an instrument.

        Args:
            instrument: Instrument identifier
            data: Historical OHLC data

        Returns:
            List of hybrid sample dictionaries
        """
        # Choose parallel or sequential processing based on worker count
        if self.workers and self.workers > 1:
            return self.create_daily_samples_parallel(instrument, data)
        else:
            return self.create_daily_samples_sequential(instrument, data)

    def create_daily_samples_sequential(self, instrument: str, data: pd.DataFrame) -> List[Dict]:
        """
        Create 4-class hybrid samples sequentially.

        Args:
            instrument: Instrument identifier
            data: Historical OHLC data

        Returns:
            List of sample dictionaries
        """
        samples = []
        errors = []

        # Need at least 2 days (current + previous) and enough for chart window
        chart_window = self.config["data"]["bars_per_chart"]
        min_data_needed = chart_window + 1  # +1 for previous day

        if len(data) < min_data_needed:
            self.logger.warning(
                f"Insufficient data for {instrument}: {len(data)} bars (need {min_data_needed})"
            )
            return []

        # Start from second day (need previous day for levels)
        for i in range(1, len(data)):
            try:
                current_date = data.index[i]
                
                # Get chart window for current analysis
                start_idx = max(0, i - chart_window + 1)
                chart_data = data.iloc[start_idx:i+1]
                
                # Get previous day data
                previous_candle = data.iloc[i-1]
                current_candle = data.iloc[i]

                # Analyze daily pattern
                pattern = self.pattern_analyzer.analyze_daily_pattern(current_candle, previous_candle)

                # Generate chart with previous day levels
                chart_path = self.chart_generator.generate_daily_chart_image(
                    current_date, chart_data, previous_candle, instrument
                )

                # Extract features
                features = self.pattern_analyzer.extract_candle_features(current_candle, previous_candle)
                numerical_features = self.pattern_analyzer.extract_numerical_features(current_candle, previous_candle)

                # Create hybrid sample
                sample = {
                    "sample_id": f"{instrument}_{current_date.strftime('%Y%m%d')}",
                    "date": current_date.isoformat(),
                    "instrument": instrument,
                    "pattern": pattern,
                    "pattern_label": {1: "High Breakout", 2: "Low Breakdown", 3: "Range Expansion", 4: "Range Bound"}[pattern],
                    "chart_path": chart_path,
                    "features": features,
                    "numerical_features": numerical_features,
                    "current_ohlc": {
                        "open": float(current_candle["Open"]),
                        "high": float(current_candle["High"]),
                        "low": float(current_candle["Low"]),
                        "close": float(current_candle["Close"]),
                    },
                    "previous_levels": {
                        "high": float(previous_candle["High"]),
                        "low": float(previous_candle["Low"]),
                    },
                }

                samples.append(sample)

                if len(samples) % 100 == 0:
                    self.logger.info(f"Created {len(samples)} samples for {instrument}")

            except Exception as e:
                error_msg = f"Error creating sample for {instrument} on {current_date}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        self.logger.info(
            f"Sequential processing complete for {instrument}: {len(samples)} samples, {len(errors)} errors"
        )

        return samples

    def create_daily_samples_parallel(self, instrument: str, data: pd.DataFrame) -> List[Dict]:
        """
        Create 4-class hybrid samples using parallel processing.

        Args:
            instrument: Instrument identifier
            data: Historical OHLC data

        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # Prepare arguments for parallel processing
        chart_window = self.config["data"]["bars_per_chart"]
        min_data_needed = chart_window + 1
        
        if len(data) < min_data_needed:
            self.logger.warning(
                f"Insufficient data for {instrument}: {len(data)} bars (need {min_data_needed})"
            )
            return []

        # Create args for each sample (need current + previous day data)
        parallel_args = []
        for i in range(1, len(data)):  # Start from second day
            current_date = data.index[i]
            
            # Get chart window data
            start_idx = max(0, i - chart_window + 1)
            chart_data = data.iloc[start_idx:i+1]
            
            # Get previous day data (single row)
            previous_data = data.iloc[i-1:i]
            
            parallel_args.append((
                current_date,
                instrument,
                chart_data,
                previous_data,
                self.config_path
            ))

        self.logger.info(f"Starting parallel processing for {instrument}: {len(parallel_args)} samples with {self.workers} workers")
        
        # Process samples in parallel
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            try:
                # Submit all tasks
                future_to_args = {
                    executor.submit(process_sample_parallel, args): args
                    for args in parallel_args
                }
                
                # Collect results with progress tracking
                completed = 0
                for future in concurrent.futures.as_completed(future_to_args):
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per sample
                        if result.get("success", False):
                            samples.append(result)
                        else:
                            self.logger.warning(f"Sample failed: {result.get('error', 'Unknown error')}")
                        
                        completed += 1
                        if completed % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            self.logger.info(f"Processed {completed}/{len(parallel_args)} samples ({rate:.1f} samples/sec)")
                            
                    except concurrent.futures.TimeoutError:
                        self.logger.error(f"Sample processing timeout for {instrument}")
                    except Exception as e:
                        self.logger.error(f"Error processing sample: {e}")
                        
            except KeyboardInterrupt:
                self.logger.warning("Parallel processing interrupted by user")
                executor.shutdown(wait=False)
                raise
                
        elapsed_time = time.time() - start_time
        processing_rate = len(samples) / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(
            f"Parallel processing complete for {instrument}: "
            f"{len(samples)} samples in {elapsed_time:.1f}s ({processing_rate:.1f} samples/sec)"
        )

        return samples

    def create_multi_instrument_dataset(
        self,
        instruments: List[str] = None,
        start_year: int = None,
        end_year: int = None,
    ) -> Dict:
        """
        Create hybrid dataset for multiple instruments with 4-class classification.

        Args:
            instruments: List of instruments to process (or None for config default)
            start_year: Start year (or None for config default)
            end_year: End year (or None for config default)

        Returns:
            Dictionary with dataset creation results
        """
        try:
            # Resolve parameters
            instruments = self.resolve_instruments(instruments)
            start_year = start_year or self.config["data"]["start_year"]
            end_year = end_year or self.config["data"]["end_year"]

            self.logger.info(f"Creating daily hybrid dataset for {instruments} ({start_year}-{end_year})")
            self.logger.info(f"Model type: 4-class hybrid (visual + numerical features)")

            total_samples = 0
            all_patterns = []
            
            # Process each instrument
            for instrument in instruments:
                self.logger.info(f"Processing {instrument}...")
                
                # Fetch instrument data
                data = self.fetch_instrument_data(instrument, start_year, end_year)
                
                # Create samples for this instrument
                samples = self.create_daily_samples(instrument, data)
                
                # Update dataset manifest
                self.dataset_manifest["instruments"][instrument] = {
                    "samples_count": len(samples),
                    "data_range": {
                        "start": data.index[0].isoformat(),
                        "end": data.index[-1].isoformat(),
                        "total_bars": len(data)
                    }
                }
                
                # Add to global collections
                self.dataset_manifest["samples"].extend(samples)
                patterns = [s["pattern"] for s in samples]
                all_patterns.extend(patterns)
                total_samples += len(samples)
                
                self.logger.info(f"Completed {instrument}: {len(samples)} samples")

            # Calculate dataset statistics
            self.calculate_dataset_statistics(all_patterns)
            
            # Save dataset manifest
            self.save_dataset_manifest()
            
            result = {
                "success": True,
                "total_samples": total_samples,
                "instruments_processed": len(instruments),
                "instruments": instruments,
                "date_range": f"{start_year}-{end_year}",
                "manifest_path": self.get_manifest_path(),
                "model_type": "4-class hybrid",
                "features": {
                    "visual": "30-bar charts with previous day levels",
                    "numerical": 3
                }
            }
            
            self.logger.info(f"Dataset creation completed successfully: {total_samples} total samples")
            return result
            
        except Exception as e:
            self.logger.error(f"Dataset creation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def calculate_dataset_statistics(self, patterns: List[int]):
        """Calculate and store 4-class pattern distribution statistics."""
        if not patterns:
            return
            
        # Count patterns (1-4)
        pattern_counts = {}
        for i in range(1, 5):
            pattern_counts[i] = patterns.count(i)
        
        total = len(patterns)
        pattern_percentages = {
            i: (count / total) * 100 for i, count in pattern_counts.items()
        }
        
        self.dataset_manifest["statistics"] = {
            "total_samples": total,
            "pattern_counts": pattern_counts,
            "pattern_percentages": pattern_percentages,
            "pattern_labels": {
                1: "High Breakout",
                2: "Low Breakdown", 
                3: "Range Expansion",
                4: "Range Bound"
            },
            "class_balance": {
                "most_common": max(pattern_counts, key=pattern_counts.get),
                "least_common": min(pattern_counts, key=pattern_counts.get),
                "balance_ratio": min(pattern_counts.values()) / max(pattern_counts.values()) if max(pattern_counts.values()) > 0 else 0
            }
        }

    def get_manifest_path(self) -> str:
        """Get path for dataset manifest file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            self.config["paths"]["metadata"],
            f"daily_hybrid_dataset_manifest_{timestamp}.json"
        )

    def save_dataset_manifest(self):
        """Save the dataset manifest to JSON file."""
        manifest_path = self.get_manifest_path()
        
        with open(manifest_path, 'w') as f:
            json.dump(self.dataset_manifest, f, indent=2, default=str)
        
        self.logger.info(f"Dataset manifest saved: {manifest_path}")

    def save_individual_sample(self, sample: Dict, instrument: str, analysis_date: datetime):
        """Save individual hybrid sample to JSON file."""
        date_str = analysis_date.strftime("%Y%m%d")
        filename = f"{self.config['paths']['labels']}/daily_{instrument}_{date_str}.json"

        with open(filename, "w") as f:
            json.dump(sample, f, indent=2, default=str)

        self.logger.debug(f"Saved individual sample: {filename}")

    def create_dataset_summary(self, instruments: List[str], total_samples: int) -> Dict:
        """Create comprehensive 4-class hybrid dataset summary."""
        summary = {
            "dataset_info": {
                "name": "4-class hybrid daily dataset",
                "model_type": "hybrid_visual_numerical",
                "total_samples": total_samples,
                "generation_date": datetime.now().isoformat(),
                "instruments": instruments,
                "classification_system": "4-class previous day levels",
            },
            "pattern_system": {
                "classes": 4,
                "labels": {
                    1: "High Breakout",
                    2: "Low Breakdown", 
                    3: "Range Expansion",
                    4: "Range Bound"
                },
                "description": "Based on current day high/low vs previous day high/low levels"
            },
            "features": {
                "visual": {
                    "type": "30-bar daily candlestick charts",
                    "reference_lines": "Previous day high (green) and low (red) levels",
                    "image_size": "224x224 pixels for ViT",
                    "format": "PNG"
                },
                "numerical": {
                    "count": 3,
                    "features": [
                        "distance_to_prev_high",
                        "distance_to_prev_low", 
                        "prev_day_range"
                    ],
                    "description": "Key metrics for hybrid model fusion"
                }
            },
            "configuration": {
                "chart_bars": self.config["data"]["bars_per_chart"],
                "classification_classes": 4,
                "hybrid_model": True,
                "parallel_processing": self.workers is not None and self.workers > 1
            },
        }

        return summary


def main():
    """Test the daily hybrid dataset creator."""
    print("🚀 NQ_AI Daily Hybrid Dataset Creator")
    print("4-Class Previous Day Levels Classification System")

    try:
        # Initialize creator with test configuration
        creator = DailyDatasetCreator()

        print("🏗️  Initializing daily hybrid dataset creator...")
        print(f"   📊 Model Type: 4-class hybrid (visual + numerical)")
        print(f"   📈 Chart Features: 30-bar charts with previous day level lines")
        print(f"   🔢 Numerical Features: 3 key metrics for fusion")
        print(f"   🏭 Parallel Processing: {'Enabled' if creator.workers else 'Disabled'}")

        # Test with a small dataset
        print("\n🧪 Testing with single instrument (NASDAQ)...")
        test_instruments = ["NASDAQ"]
        
        result = creator.create_multi_instrument_dataset(
            instruments=test_instruments,
            start_year=2024,  # Small test range
            end_year=2024
        )

        if result["success"]:
            print("✅ Test dataset created successfully!")
            print(f"   📈 Total samples: {result['total_samples']}")
            print(f"   🎯 Model type: {result['model_type']}")
            print(f"   📊 Visual features: {result['features']['visual']}")
            print(f"   🔢 Numerical features: {result['features']['numerical']}")
            print(f"   📁 Manifest: {result['manifest_path']}")
            
            # Show pattern distribution if available
            if creator.dataset_manifest.get("statistics"):
                stats = creator.dataset_manifest["statistics"]
                print("   🎯 Pattern Distribution:")
                for pattern_id, count in stats["pattern_counts"].items():
                    label = stats["pattern_labels"][pattern_id]
                    percentage = stats["pattern_percentages"][pattern_id]
                    print(f"      {pattern_id}: {label} - {count} ({percentage:.1f}%)")
        else:
            print(f"❌ Test failed: {result['error']}")
            return 1

        print("\n🎯 Daily Hybrid Dataset Features:")
        print("   - 4-class previous day levels classification")
        print("   - Hybrid model: visual charts + numerical features")
        print("   - Previous day level reference lines (green high, red low)")
        print("   - Multi-instrument support (DOW, NASDAQ, SP500)")
        print("   - Parallel processing for fast generation")
        print("   - 30-bar chart windows (approximately 6 weeks context)")
        print("   - ViT-ready 224x224 pixel images")
        print("   - 3 numerical features for fusion architecture")

        print("\n✅ Daily hybrid dataset creator test completed successfully!")
        print("\n🚀 Ready for full dataset generation:")
        print("   python generate_daily_dataset.py --instruments ALL --parallel")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
