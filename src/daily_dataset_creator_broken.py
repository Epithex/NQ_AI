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
        """
        Generate complete daily pattern dataset.
        
        Args:
            start_year: Starting year (default from config)
            end_year: Ending year (default from config)
            dataset_name: Name for the dataset
        """
        if start_year is None:
            start_year = self.config['data']['start_year']
        if end_year is None:
            end_year = self.config['data']['end_year']
            
        self.logger.info(f"=" * 60)
        self.logger.info(f"Starting daily dataset generation: {dataset_name}")
        self.logger.info(f"Period: {start_year}-{end_year}")
        self.logger.info(f"=" * 60)
        
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
            self.logger.info(f"✅ Dataset summary created")
            
            # Step 4: Save complete dataset metadata
            self.logger.info("Step 4: Saving dataset metadata...")
            self.save_dataset_metadata(samples, summary, dataset_name)
            self.logger.info(f"✅ Metadata saved")
            
            self.logger.info(f"=" * 60)
            self.logger.info(f"DATASET GENERATION COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Total samples: {len(samples)}")
            self.logger.info(f"Date range: {samples[0]['date']} to {samples[-1]['date']}")
            self.logger.info(f"Chart images: {self.config['paths']['images']}")
            self.logger.info(f"Sample labels: {self.config['paths']['labels']}")
            self.logger.info(f"Metadata: {self.config['paths']['metadata']}")
            self.logger.info(f"=" * 60)
            
            return samples, summary
            
        except Exception as e:
            self.logger.error(f"Fatal error in dataset generation: {str(e)}")
            raise
    
    def process_all_trading_days(self, daily_data: pd.DataFrame) -> List[Dict]:
        """
        Process all trading days to generate samples.
        
        Args:
            daily_data: Full daily price data
            
        Returns:
            List of training samples
        """
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
                    if processed % 100 == 0:
                        self.logger.info(f"Progress: {processed}/{total_days} samples generated ({(processed/total_days)*100:.1f}%)")
                        
            except Exception as e:
                error_msg = f"Error processing {analysis_date.date()}: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        self.logger.info(f"Processing complete: {len(samples)} samples, {len(errors)} errors")\n        \n        if errors:\n            self.logger.warning(f\"Sample of errors: {errors[:3]}...\")\n        \n        return samples\n    \n    def process_single_trading_day(self, analysis_date: datetime, daily_data: pd.DataFrame, \n                                  data_index: int) -> Optional[Dict]:\n        \"\"\"\n        Process a single trading day to generate training sample.\n        \n        Args:\n            analysis_date: Date being analyzed\n            daily_data: Full daily price data\n            data_index: Index position in data\n            \n        Returns:\n            Training sample dictionary or None if processing fails\n        \"\"\"\n        try:\n            # Get the daily candle being analyzed\n            daily_candle = daily_data.iloc[data_index]\n            \n            # Get previous day levels\n            prev_day = daily_data.iloc[data_index - 1]\n            prev_high = float(prev_day['High'])\n            prev_low = float(prev_day['Low'])\n            \n            # Generate chart image (30-bar window)\n            chart_path = self.chart_generator.generate_chart_image(analysis_date, daily_data)\n            \n            # Analyze pattern\n            pattern_rank = self.pattern_analyzer.analyze_daily_pattern(daily_candle, prev_high, prev_low)\n            \n            # Extract numerical features\n            features = self.pattern_analyzer.extract_numerical_features(daily_candle, prev_high, prev_low)\n            \n            # Create complete sample\n            sample = {\n                'date': analysis_date.strftime('%Y-%m-%d'),\n                'timestamp': analysis_date.isoformat(),\n                'chart_image': chart_path,\n                'pattern_rank': pattern_rank,\n                'pattern_label': self.config['classification']['labels'][pattern_rank],\n                'features': features,\n                'previous_levels': {\n                    'high': prev_high,\n                    'low': prev_low,\n                    'range': prev_high - prev_low\n                },\n                'daily_candle': {\n                    'open': float(daily_candle['Open']),\n                    'high': float(daily_candle['High']),\n                    'low': float(daily_candle['Low']),\n                    'close': float(daily_candle['Close']),\n                    'volume': float(daily_candle['Volume']) if 'Volume' in daily_candle else 0\n                },\n                'metadata': {\n                    'chart_bars': self.config['data']['bars_per_chart'],\n                    'system_version': self.config['system']['version'],\n                    'generation_timestamp': datetime.now().isoformat()\n                }\n            }\n            \n            # Save individual sample\n            self.save_individual_sample(sample, analysis_date)\n            \n            return sample\n            \n        except Exception as e:\n            self.logger.error(f\"Error processing {analysis_date}: {e}\")\n            return None\n    \n    def save_individual_sample(self, sample: Dict, analysis_date: datetime):\n        \"\"\"\n        Save individual sample to JSON file.\n        \n        Args:\n            sample: Training sample data\n            analysis_date: Analysis date for filename\n        \"\"\"\n        date_str = analysis_date.strftime('%Y%m%d')\n        filename = f\"{self.config['paths']['labels']}/daily_sample_{date_str}.json\"\n        \n        with open(filename, 'w') as f:\n            json.dump(sample, f, indent=2, default=str)\n    \n    def create_dataset_summary(self, samples: List[Dict], daily_data: pd.DataFrame, \n                              dataset_name: str) -> Dict:\n        \"\"\"\n        Create comprehensive dataset summary.\n        \n        Args:\n            samples: Generated training samples\n            daily_data: Source daily data\n            dataset_name: Name of the dataset\n            \n        Returns:\n            Dataset summary dictionary\n        \"\"\"\n        # Calculate pattern distribution\n        pattern_counts = {}\n        for i in range(1, 5):\n            pattern_counts[i] = sum(1 for s in samples if s['pattern_rank'] == i)\n        \n        # Calculate feature statistics\n        feature_stats = self.calculate_feature_statistics(samples)\n        \n        # Get date range\n        start_date = samples[0]['date'] if samples else None\n        end_date = samples[-1]['date'] if samples else None\n        \n        summary = {\n            'dataset_info': {\n                'name': dataset_name,\n                'total_samples': len(samples),\n                'generation_date': datetime.now().isoformat(),\n                'date_range': {\n                    'start': start_date,\n                    'end': end_date\n                },\n                'source_data': {\n                    'total_daily_bars': len(daily_data),\n                    'source_date_range': {\n                        'start': daily_data.index[0].strftime('%Y-%m-%d'),\n                        'end': daily_data.index[-1].strftime('%Y-%m-%d')\n                    }\n                }\n            },\n            'pattern_distribution': {\n                'counts': pattern_counts,\n                'percentages': {\n                    i: (count / len(samples)) * 100 if samples else 0\n                    for i, count in pattern_counts.items()\n                },\n                'labels': self.config['classification']['labels']\n            },\n            'feature_statistics': feature_stats,\n            'configuration': {\n                'chart_bars': self.config['data']['bars_per_chart'],\n                'classification_classes': self.config['classification']['num_classes'],\n                'feature_names': self.config['features']['feature_names'],\n                'system_version': self.config['system']['version']\n            },\n            'file_structure': {\n                'chart_images': self.config['paths']['images'],\n                'sample_labels': self.config['paths']['labels'],\n                'metadata': self.config['paths']['metadata']\n            }\n        }\n        \n        return summary\n    \n    def calculate_feature_statistics(self, samples: List[Dict]) -> Dict:\n        \"\"\"\n        Calculate statistics for numerical features.\n        \n        Args:\n            samples: Training samples\n            \n        Returns:\n            Feature statistics dictionary\n        \"\"\"\n        if not samples:\n            return {}\n        \n        feature_names = self.config['features']['feature_names']\n        feature_stats = {}\n        \n        for feature_name in feature_names:\n            values = [s['features'][feature_name] for s in samples if feature_name in s['features']]\n            \n            if values:\n                feature_stats[feature_name] = {\n                    'count': len(values),\n                    'mean': sum(values) / len(values),\n                    'min': min(values),\n                    'max': max(values),\n                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5\n                }\n            else:\n                feature_stats[feature_name] = {'count': 0}\n        \n        return feature_stats\n    \n    def save_dataset_metadata(self, samples: List[Dict], summary: Dict, dataset_name: str):\n        \"\"\"\n        Save complete dataset metadata.\n        \n        Args:\n            samples: Generated samples\n            summary: Dataset summary\n            dataset_name: Name of the dataset\n        \"\"\"\n        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n        \n        # Save dataset summary\n        summary_file = f\"{self.config['paths']['metadata']}/{dataset_name}_summary_{timestamp}.json\"\n        with open(summary_file, 'w') as f:\n            json.dump(summary, f, indent=2, default=str)\n        \n        # Save sample index (list of all samples)\n        sample_index = [\n            {\n                'date': s['date'],\n                'pattern': s['pattern_rank'],\n                'chart_image': s['chart_image'],\n                'label_file': f\"{self.config['paths']['labels']}/daily_sample_{s['date'].replace('-', '')}.json\"\n            }\n            for s in samples\n        ]\n        \n        index_file = f\"{self.config['paths']['metadata']}/{dataset_name}_index_{timestamp}.json\"\n        with open(index_file, 'w') as f:\n            json.dump(sample_index, f, indent=2)\n        \n        # Save pattern distribution for quick reference\n        distribution_file = f\"{self.config['paths']['metadata']}/{dataset_name}_distribution_{timestamp}.json\"\n        with open(distribution_file, 'w') as f:\n            json.dump(summary['pattern_distribution'], f, indent=2)\n        \n        self.logger.info(f\"Metadata saved:\")\n        self.logger.info(f\"  Summary: {summary_file}\")\n        self.logger.info(f\"  Index: {index_file}\")\n        self.logger.info(f\"  Distribution: {distribution_file}\")\n    \n    def load_existing_dataset(self, dataset_name: str) -> Tuple[List[Dict], Dict]:\n        \"\"\"\n        Load existing dataset from files.\n        \n        Args:\n            dataset_name: Name of the dataset to load\n            \n        Returns:\n            Tuple of (samples, summary)\n        \"\"\"\n        # Find most recent dataset files\n        metadata_dir = Path(self.config['paths']['metadata'])\n        \n        # Find summary file\n        summary_files = list(metadata_dir.glob(f\"{dataset_name}_summary_*.json\"))\n        if not summary_files:\n            raise FileNotFoundError(f\"No summary file found for dataset: {dataset_name}\")\n        \n        latest_summary = max(summary_files, key=os.path.getctime)\n        \n        # Load summary\n        with open(latest_summary, 'r') as f:\n            summary = json.load(f)\n        \n        # Load samples from individual files\n        samples = []\n        labels_dir = Path(self.config['paths']['labels'])\n        \n        for sample_file in sorted(labels_dir.glob(\"daily_sample_*.json\")):\n            with open(sample_file, 'r') as f:\n                sample = json.load(f)\n                samples.append(sample)\n        \n        self.logger.info(f\"Loaded existing dataset: {len(samples)} samples\")\n        return samples, summary\n\ndef main():\n    \"\"\"Test the daily dataset creator.\"\"\"\n    print(\"Testing Daily Dataset Creator...\")\n    \n    try:\n        # Initialize creator\n        creator = DailyDatasetCreator()\n        \n        # Generate small test dataset (recent data only)\n        print(\"📊 Generating test dataset...\")\n        samples, summary = creator.generate_dataset(\n            start_year=2024,\n            end_year=2025,\n            dataset_name=\"test_daily_nq\"\n        )\n        \n        print(f\"✅ Dataset generated successfully!\")\n        print(f\"   📈 Total samples: {len(samples)}\")\n        print(f\"   📅 Date range: {samples[0]['date']} to {samples[-1]['date']}\")\n        print(f\"   🎯 Pattern distribution:\")\n        \n        for pattern, count in summary['pattern_distribution']['counts'].items():\n            label = summary['pattern_distribution']['labels'][pattern]\n            percentage = summary['pattern_distribution']['percentages'][pattern]\n            print(f\"      {pattern}: {label} - {count} ({percentage:.1f}%)\")\n        \n        print(f\"   📁 Files saved to:\")\n        print(f\"      Charts: {creator.config['paths']['images']}\")\n        print(f\"      Labels: {creator.config['paths']['labels']}\")\n        print(f\"      Metadata: {creator.config['paths']['metadata']}\")\n        \n        return 0\n        \n    except Exception as e:\n        print(f\"❌ Error: {e}\")\n        return 1\n\nif __name__ == \"__main__\":\n    exit(main())")