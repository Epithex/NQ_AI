# src/dataset_creator.py
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import pytz
import yaml
import logging
from typing import Dict, List
from .data_fetcher import DataFetcher
from .chart_generator import ChartGenerator
from .session_analyzer import SessionAnalyzer

class DatasetCreator:
    """Orchestrates the complete data generation pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DatasetCreator with all components."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_fetcher = DataFetcher(config_path)
        self.chart_generator = ChartGenerator(config_path)
        self.session_analyzer = SessionAnalyzer(config_path)
        
        self.timezone = pytz.timezone(self.config['time']['timezone'])
        self.paths = self.config['paths']
        
        # Setup logging
        logging.basicConfig(
            filename=f"{self.config['paths']['logs']}/dataset_creator.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    
    def generate_dataset(self, start_date: str, end_date: str, dataset_type: str = "training"):
        """
        Main method to generate complete dataset.
        
        Args:
            start_date: Start date for dataset generation
            end_date: End date for dataset generation
            dataset_type: Type of dataset (training/validation/test)
        """
        try:
            self.logger.info(f"Starting dataset generation from {start_date} to {end_date}")
            
            # Fetch all historical data with buffer
            all_data = self.data_fetcher.fetch_data(start_date, end_date)
            
            # Validate data quality
            if not self.data_fetcher.validate_data_quality(all_data):
                raise ValueError("Data quality validation failed")
            
            # Process each trading day
            current_date = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            samples_generated = 0
            errors = []
            
            while current_date <= end_dt:
                # Skip weekends
                if current_date.weekday() in [5, 6]:
                    current_date += timedelta(days=1)
                    continue
                
                try:
                    # Process single trading day
                    sample = self.process_trading_day(current_date, all_data)
                    if sample:
                        samples_generated += 1
                        self.logger.info(f"Generated sample {samples_generated} for {current_date.date()}")
                    
                except Exception as e:
                    error_msg = f"Error processing {current_date.date()}: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                
                current_date += timedelta(days=1)
            
            # Generate dataset summary
            self.create_dataset_summary(samples_generated, errors, dataset_type)
            
            self.logger.info(f"Dataset generation complete: {samples_generated} samples created")
            
        except Exception as e:
            self.logger.error(f"Fatal error in dataset generation: {str(e)}")
            raise
    
    def process_trading_day(self, trading_date: datetime, all_data: pd.DataFrame) -> Dict:
        """
        Process a single trading day to generate training sample.
        
        Args:
            trading_date: Date to process
            all_data: All historical data
            
        Returns:
            Dictionary with sample data or None if day cannot be processed
        """
        try:
            # Set analysis time (7 AM EST)
            analysis_time = self.timezone.localize(
                trading_date.replace(hour=7, minute=0, second=0)
            )
            
            # Get previous day levels
            prev_high, prev_low = self.data_fetcher.get_previous_day_levels(
                trading_date, all_data
            )
            
            # Get 300 bars for chart
            chart_data = self.data_fetcher.get_data_for_chart(
                analysis_time, all_data, bars=300
            )
            
            # Validate chart data
            if not self.chart_generator.validate_chart_data(chart_data):
                self.logger.warning(f"Invalid chart data for {trading_date.date()}")
                return None
            
            # Generate chart image
            chart_path = self.chart_generator.generate_chart_image(
                chart_data, prev_high, prev_low, analysis_time
            )
            
            # Get session data
            session_data = self.session_analyzer.get_session_data(
                trading_date, all_data
            )
            
            # Analyze session for classification
            actual_label = self.session_analyzer.analyze_session(
                session_data, prev_high, prev_low
            )
            
            # Generate success/failure matrix
            success_failure = self.session_analyzer.determine_success_failure(actual_label)
            
            # Get touch details
            touch_details = self.session_analyzer.get_touch_details(
                session_data, prev_high, prev_low
            )
            
            # Create training sample
            sample = self.create_training_sample(
                trading_date=trading_date,
                chart_path=chart_path,
                actual_label=actual_label,
                success_failure=success_failure,
                prev_high=prev_high,
                prev_low=prev_low,
                touch_details=touch_details
            )
            
            # Save label data
            self.save_label_data(sample, trading_date)
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error processing {trading_date.date()}: {str(e)}")
            raise
    
    def create_training_sample(self, trading_date: datetime, chart_path: str, 
                              actual_label: int, success_failure: Dict,
                              prev_high: float, prev_low: float,
                              touch_details: Dict) -> Dict:
        """
        Create complete training sample with all metadata.
        
        Args:
            trading_date: Trading date
            chart_path: Path to chart image
            actual_label: Actual classification (1-6)
            success_failure: Success/failure for each label
            prev_high: Previous day high
            prev_low: Previous day low
            touch_details: Detailed touch information
            
        Returns:
            Complete training sample dictionary
        """
        sample = {
            'date': trading_date.strftime('%Y-%m-%d'),
            'timestamp': trading_date.strftime('%Y-%m-%d %H:%M:%S'),
            'chart_image': chart_path,
            'actual_label': actual_label,
            'labels': success_failure,
            'previous_levels': {
                'high': prev_high,
                'low': prev_low
            },
            'touch_details': touch_details,
            'metadata': {
                'analysis_time': '07:00 EST',
                'session_start': '08:00 EST',
                'session_end': '17:00 EST',
                'chart_bars': 300
            }
        }
        
        return sample
    
    def save_label_data(self, sample: Dict, trading_date: datetime):
        """
        Save label data to JSON file.
        
        Args:
            sample: Training sample data
            trading_date: Trading date for filename
        """
        date_str = trading_date.strftime('%Y%m%d')
        label_file = f"{self.paths['labels']}/label_{date_str}.json"
        
        with open(label_file, 'w') as f:
            json.dump(sample, f, indent=2, default=str)
        
        self.logger.info(f"Label data saved: {label_file}")
    
    def create_dataset_summary(self, samples_generated: int, errors: List[str], 
                              dataset_type: str):
        """
        Create summary file for the dataset.
        
        Args:
            samples_generated: Number of samples created
            errors: List of errors encountered
            dataset_type: Type of dataset
        """
        summary = {
            'dataset_type': dataset_type,
            'samples_generated': samples_generated,
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'errors': errors,
            'label_distribution': self.calculate_label_distribution()
        }
        
        summary_file = f"{self.paths['metadata']}/dataset_summary_{dataset_type}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Dataset summary saved: {summary_file}")
    
    def calculate_label_distribution(self) -> Dict[int, int]:
        """
        Calculate distribution of labels in generated dataset.
        
        Returns:
            Dictionary with label counts
        """
        distribution = {i: 0 for i in range(1, 7)}
        
        # Count labels from saved files
        label_files = [f for f in os.listdir(self.paths['labels']) if f.endswith('.json')]
        
        for file in label_files:
            try:
                with open(f"{self.paths['labels']}/{file}", 'r') as f:
                    data = json.load(f)
                    label = data['actual_label']
                    distribution[label] += 1
            except Exception as e:
                self.logger.warning(f"Error reading label file {file}: {str(e)}")
        
        return distribution