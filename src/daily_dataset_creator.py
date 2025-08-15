#!/usr/bin/env python3
"""
Daily Dataset Creator for NQ Pattern Analysis
Orchestrates the complete data generation pipeline for daily NQ analysis
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging

# Import our components
from stooq_fetcher import DailyNQFetcher
from daily_chart_generator import DailyChartGenerator
from daily_pattern_analyzer import DailyPatternAnalyzer

class DailyDatasetCreator:
    """Creates daily NQ pattern analysis dataset."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize dataset creator with all components."""
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.data_fetcher = DailyNQFetcher(config_path)
        self.chart_generator = DailyChartGenerator(config_path)
        self.pattern_analyzer = DailyPatternAnalyzer(config_path)
        
        self.setup_logging()
        self.ensure_directories()
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging for the dataset creator."""
        log_dir = self.config['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/dataset_creator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        for path_key, path_value in self.config['paths'].items():
            if path_key != 'logs_dir':  # Already created in setup_logging
                os.makedirs(path_value, exist_ok=True)
    
    def generate_dataset(self, start_year: int = None, end_year: int = None, 
                        dataset_name: str = "daily_nq_patterns"):
        """Generate complete daily pattern dataset."""
        if start_year is None:
            start_year = self.config['data']['start_year']
        if end_year is None:
            end_year = self.config['data']['end_year']
            
        self.logger.info("=" * 60)
        self.logger.info(f"Starting daily dataset generation: {dataset_name}")
        self.logger.info(f"Period: {start_year}-{end_year}")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Fetch all daily data
            self.logger.info("Step 1: Fetching daily NQ data...")
            daily_data = self.data_fetcher.fetch_daily_data(start_year, end_year)
            self.logger.info(f"✅ Fetched {len(daily_data)} daily bars")
            
            # Step 2: Process each trading day
            self.logger.info("Step 2: Processing trading days...")
            samples = self.process_all_trading_days(daily_data)
            self.logger.info(f"✅ Generated {len(samples)} samples")
            
            # Step 3: Generate dataset summary
            self.logger.info("Step 3: Creating dataset summary...")
            summary = self.create_dataset_summary(samples, daily_data, dataset_name)
            self.logger.info("✅ Dataset summary created")
            
            # Step 4: Save complete dataset metadata
            self.logger.info("Step 4: Saving dataset metadata...")
            self.save_dataset_metadata(samples, summary, dataset_name)
            self.logger.info("✅ Metadata saved")
            
            self.logger.info("=" * 60)
            self.logger.info("DATASET GENERATION COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Total samples: {len(samples)}")
            if samples:
                self.logger.info(f"Date range: {samples[0]['date']} to {samples[-1]['date']}")
            self.logger.info("=" * 60)
            
            return samples, summary
            
        except Exception as e:
            self.logger.error(f"Fatal error in dataset generation: {str(e)}")
            raise
    
    def process_all_trading_days(self, daily_data: pd.DataFrame) -> List[Dict]:
        """Process all trading days to generate samples."""
        samples = []
        errors = []
        
        # Start processing after we have enough bars for charts
        min_bars = self.config['data']['bars_per_chart']
        start_idx = min_bars
        
        total_days = len(daily_data) - start_idx
        self.logger.info(f"Processing {total_days} trading days (starting from index {start_idx})")
        
        for i in range(start_idx, len(daily_data)):
            analysis_date = daily_data.index[i]
            
            try:
                sample = self.process_single_trading_day(analysis_date, daily_data, i)
                if sample:
                    samples.append(sample)
                    
                    # Progress logging
                    processed = len(samples)
                    if processed % 50 == 0:
                        progress_pct = (processed/total_days)*100
                        self.logger.info(f"Progress: {processed}/{total_days} samples ({progress_pct:.1f}%)")
                        
            except Exception as e:
                error_msg = f"Error processing {analysis_date.date()}: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        self.logger.info(f"Processing complete: {len(samples)} samples, {len(errors)} errors")
        
        if errors:
            self.logger.warning(f"Sample of errors: {errors[:3]}...")
        
        return samples
    
    def process_single_trading_day(self, analysis_date: datetime, daily_data: pd.DataFrame, 
                                  data_index: int) -> Optional[Dict]:
        """Process a single trading day to generate training sample."""
        try:
            # Get the daily candle being analyzed
            daily_candle = daily_data.iloc[data_index]
            
            # Get previous day levels
            prev_day = daily_data.iloc[data_index - 1]
            prev_high = float(prev_day['High'])
            prev_low = float(prev_day['Low'])
            
            # Generate chart image (30-bar window)
            chart_path = self.chart_generator.generate_chart_image(analysis_date, daily_data)
            
            # Analyze pattern
            pattern_rank = self.pattern_analyzer.analyze_daily_pattern(daily_candle, prev_high, prev_low)
            
            # Extract numerical features
            features = self.pattern_analyzer.extract_numerical_features(daily_candle, prev_high, prev_low)
            
            # Create complete sample
            sample = {
                'date': analysis_date.strftime('%Y-%m-%d'),
                'timestamp': analysis_date.isoformat(),
                'chart_image': chart_path,
                'pattern_rank': pattern_rank,
                'pattern_label': self.config['classification']['labels'][pattern_rank],
                'features': features,
                'previous_levels': {
                    'high': prev_high,
                    'low': prev_low,
                    'range': prev_high - prev_low
                },
                'daily_candle': {
                    'open': float(daily_candle['Open']),
                    'high': float(daily_candle['High']),
                    'low': float(daily_candle['Low']),
                    'close': float(daily_candle['Close']),
                    'volume': float(daily_candle['Volume']) if 'Volume' in daily_candle else 0
                },
                'metadata': {
                    'chart_bars': self.config['data']['bars_per_chart'],
                    'system_version': self.config['system']['version'],
                    'generation_timestamp': datetime.now().isoformat()
                }
            }
            
            # Save individual sample
            self.save_individual_sample(sample, analysis_date)
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error processing {analysis_date}: {e}")
            return None
    
    def save_individual_sample(self, sample: Dict, analysis_date: datetime):
        """Save individual sample to JSON file."""
        date_str = analysis_date.strftime('%Y%m%d')
        filename = f"{self.config['paths']['labels']}/daily_sample_{date_str}.json"
        
        with open(filename, 'w') as f:
            json.dump(sample, f, indent=2, default=str)
    
    def create_dataset_summary(self, samples: List[Dict], daily_data: pd.DataFrame, 
                              dataset_name: str) -> Dict:
        """Create comprehensive dataset summary."""
        # Calculate pattern distribution
        pattern_counts = {}
        for i in range(1, 5):
            pattern_counts[i] = sum(1 for s in samples if s['pattern_rank'] == i)
        
        # Get date range
        start_date = samples[0]['date'] if samples else None
        end_date = samples[-1]['date'] if samples else None
        
        summary = {
            'dataset_info': {
                'name': dataset_name,
                'total_samples': len(samples),
                'generation_date': datetime.now().isoformat(),
                'date_range': {
                    'start': start_date,
                    'end': end_date
                }
            },
            'pattern_distribution': {
                'counts': pattern_counts,
                'percentages': {
                    i: (count / len(samples)) * 100 if samples else 0
                    for i, count in pattern_counts.items()
                },
                'labels': self.config['classification']['labels']
            },
            'configuration': {
                'chart_bars': self.config['data']['bars_per_chart'],
                'classification_classes': self.config['classification']['num_classes'],
                'system_version': self.config['system']['version']
            }
        }
        
        return summary
    
    def save_dataset_metadata(self, samples: List[Dict], summary: Dict, dataset_name: str):
        """Save complete dataset metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save dataset summary
        summary_file = f"{self.config['paths']['metadata']}/{dataset_name}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Metadata saved: {summary_file}")

def main():
    """Generate complete 25-year daily dataset."""
    print("🚀 NQ_AI Daily Dataset Generation")
    print("Generating complete 25-year dataset (2000-2025)")
    
    try:
        # Initialize creator
        creator = DailyDatasetCreator()
        
        # Generate complete 25-year dataset using config defaults
        print("📊 Generating complete 25-year dataset...")
        samples, summary = creator.generate_dataset(
            dataset_name="daily_nq_25yr"
        )
        
        print("✅ Dataset generated successfully!")
        print(f"   📈 Total samples: {len(samples)}")
        if samples:
            print(f"   📅 Date range: {samples[0]['date']} to {samples[-1]['date']}")
        print("   🎯 Pattern distribution:")
        
        for pattern, count in summary['pattern_distribution']['counts'].items():
            label = summary['pattern_distribution']['labels'][pattern]
            percentage = summary['pattern_distribution']['percentages'][pattern]
            print(f"      {pattern}: {label} - {count} ({percentage:.1f}%)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())