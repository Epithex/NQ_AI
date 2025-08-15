#!/usr/bin/env python3
"""
Daily Data Loader for NQ Pattern Analysis
TensorFlow data pipeline for loading chart images and numerical features
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
import yaml
import logging
from datetime import datetime
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DailyDataLoader:
    """TensorFlow data loader for daily NQ pattern analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data loader."""
        self.config = self.load_config(config_path)
        self.feature_config = self.config['features']
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        
        self.setup_logging()
        
        # Data structures
        self.samples_index = None
        self.feature_scaler = None
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging for the data loader."""
        log_dir = self.config['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/data_loader.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_dataset_index(self, dataset_name: str = "daily_nq_patterns") -> List[Dict]:
        """
        Load dataset index from metadata files.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            List of sample metadata
        """
        self.logger.info(f"Loading dataset index for: {dataset_name}")
        
        # Load all individual sample files
        samples = []
        labels_dir = Path(self.paths['labels'])
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        # Get all daily sample files
        sample_files = sorted(labels_dir.glob("daily_sample_*.json"))
        
        if not sample_files:
            raise FileNotFoundError(f"No sample files found in {labels_dir}")
        
        self.logger.info(f"Found {len(sample_files)} sample files")
        
        # Load each sample
        for sample_file in sample_files:
            try:
                with open(sample_file, 'r') as f:
                    sample = json.load(f)
                    
                # Validate sample has required fields
                if self.validate_sample(sample):
                    samples.append(sample)
                else:
                    self.logger.warning(f"Invalid sample in {sample_file}")
                    
            except Exception as e:
                self.logger.error(f"Error loading {sample_file}: {e}")
        
        self.logger.info(f"Successfully loaded {len(samples)} valid samples")
        self.samples_index = samples
        return samples
    
    def validate_sample(self, sample: Dict) -> bool:
        """
        Validate sample has all required fields.
        
        Args:
            sample: Sample dictionary to validate
            
        Returns:
            True if sample is valid
        """
        required_fields = [
            'chart_image', 'pattern_rank', 'features', 
            'previous_levels', 'daily_candle'
        ]
        
        # Check required fields exist
        for field in required_fields:
            if field not in sample:
                return False
        
        # Check chart image file exists
        image_path = sample['chart_image']
        if not os.path.exists(image_path):
            self.logger.warning(f"Image file not found: {image_path}")
            return False
        
        # Check features have correct structure
        feature_names = self.feature_config['feature_names']
        for feature_name in feature_names:
            if feature_name not in sample['features']:
                return False
        
        return True
    
    def split_dataset(self, test_size: float = 0.2, val_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[List, List, List]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        if self.samples_index is None:
            raise ValueError("Dataset index not loaded. Call load_dataset_index() first.")
        
        self.logger.info(f"Splitting {len(self.samples_index)} samples...")
        
        # Create temporal split (maintain chronological order)
        samples_df = pd.DataFrame(self.samples_index)
        samples_df['date'] = pd.to_datetime(samples_df['date'])
        samples_df = samples_df.sort_values('date')
        
        total_samples = len(samples_df)
        test_split_idx = int(total_samples * (1 - test_size))
        val_split_idx = int(test_split_idx * (1 - val_size))
        
        # Split chronologically (older = train, newer = test)
        train_df = samples_df.iloc[:val_split_idx]
        val_df = samples_df.iloc[val_split_idx:test_split_idx]
        test_df = samples_df.iloc[test_split_idx:]
        
        self.train_samples = train_df.to_dict('records')
        self.val_samples = val_df.to_dict('records')
        self.test_samples = test_df.to_dict('records')
        
        self.logger.info(f"Dataset split:")
        self.logger.info(f"  Train: {len(self.train_samples)} samples")
        self.logger.info(f"  Validation: {len(self.val_samples)} samples")
        self.logger.info(f"  Test: {len(self.test_samples)} samples")
        
        # Log date ranges
        if self.train_samples:
            train_start = min(s['date'] for s in self.train_samples)
            train_end = max(s['date'] for s in self.train_samples)
            self.logger.info(f"  Train range: {train_start} to {train_end}")
        
        if self.test_samples:
            test_start = min(s['date'] for s in self.test_samples)
            test_end = max(s['date'] for s in self.test_samples)
            self.logger.info(f"  Test range: {test_start} to {test_end}")
        
        return self.train_samples, self.val_samples, self.test_samples
    
    def fit_feature_scaler(self, samples: List[Dict]) -> StandardScaler:
        """
        Fit feature scaler on training data.
        
        Args:
            samples: Training samples
            
        Returns:
            Fitted StandardScaler
        """
        feature_names = self.feature_config['feature_names']
        
        # Extract features
        features_list = []
        for sample in samples:
            features = [sample['features'][name] for name in feature_names]
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # Fit scaler
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(features_array)
        
        self.logger.info(f"Feature scaler fitted on {len(samples)} training samples")
        self.logger.info(f"Feature means: {self.feature_scaler.mean_}")
        self.logger.info(f"Feature stds: {self.feature_scaler.scale_}")
        
        return self.feature_scaler
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess chart image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image_size = self.model_config['image_size']
        image = cv2.resize(image, (image_size, image_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def preprocess_features(self, sample: Dict) -> np.ndarray:
        """
        Preprocess numerical features.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Preprocessed features array
        """
        feature_names = self.feature_config['feature_names']
        
        # Extract features
        features = [sample['features'][name] for name in feature_names]
        features_array = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Apply scaling if scaler is fitted
        if self.feature_scaler is not None:
            features_array = self.feature_scaler.transform(features_array)
        
        return features_array.flatten()
    
    def create_sample_generator(self, samples: List[Dict]) -> Generator:
        """
        Create generator for samples.
        
        Args:
            samples: List of samples
            
        Yields:
            Tuple of (image, features, label)
        """
        for sample in samples:
            try:
                # Load and preprocess image
                image = self.load_and_preprocess_image(sample['chart_image'])
                
                # Preprocess features
                features = self.preprocess_features(sample)
                
                # Get label (convert to 0-based indexing)
                label = sample['pattern_rank'] - 1
                
                yield (image, features, label)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {sample.get('date', 'unknown')}: {e}")
                continue
    
    def create_tf_dataset(self, samples: List[Dict], batch_size: int = None, 
                         shuffle: bool = True, repeat: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.
        
        Args:
            samples: List of samples
            batch_size: Batch size (default from config)
            shuffle: Whether to shuffle the dataset
            repeat: Whether to repeat the dataset
            
        Returns:
            TensorFlow dataset
        """
        if batch_size is None:
            batch_size = self.model_config['batch_size']
        
        self.logger.info(f"Creating TensorFlow dataset with {len(samples)} samples")
        
        # Define output signature for TensorFlow dataset
        output_signature = (
            tf.TensorSpec(shape=(self.model_config['image_size'], 
                               self.model_config['image_size'], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(len(self.feature_config['feature_names']),), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            lambda: self.create_sample_generator(samples),
            output_signature=output_signature
        )
        
        # Restructure dataset for multi-input model: ((image, features), label)
        dataset = dataset.map(lambda image, features, label: ((image, features), label))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, len(samples)))
        
        if repeat:
            dataset = dataset.repeat()
        
        # Batch the dataset
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_train_dataset(self, batch_size: int = None) -> tf.data.Dataset:
        """Get training dataset."""
        if not self.train_samples:
            raise ValueError("No training samples. Call split_dataset() first.")
        
        return self.create_tf_dataset(
            self.train_samples, 
            batch_size=batch_size, 
            shuffle=True, 
            repeat=False
        )
    
    def get_val_dataset(self, batch_size: int = None) -> tf.data.Dataset:
        """Get validation dataset."""
        if not self.val_samples:
            raise ValueError("No validation samples. Call split_dataset() first.")
        
        return self.create_tf_dataset(
            self.val_samples, 
            batch_size=batch_size, 
            shuffle=False, 
            repeat=False
        )
    
    def get_test_dataset(self, batch_size: int = None) -> tf.data.Dataset:
        """Get test dataset."""
        if not self.test_samples:
            raise ValueError("No test samples. Call split_dataset() first.")
        
        return self.create_tf_dataset(
            self.test_samples, 
            batch_size=batch_size, 
            shuffle=False, 
            repeat=False
        )
    
    def get_class_distribution(self, samples: List[Dict]) -> Dict[int, int]:
        """
        Get class distribution for samples.
        
        Args:
            samples: List of samples
            
        Returns:
            Dictionary with class counts
        """
        distribution = {}
        for i in range(1, 5):  # Patterns 1-4
            distribution[i] = sum(1 for s in samples if s['pattern_rank'] == i)
        
        return distribution
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for training.
        
        Returns:
            Dictionary with class weights
        """
        if not self.train_samples:
            raise ValueError("No training samples. Call split_dataset() first.")
        
        distribution = self.get_class_distribution(self.train_samples)
        total_samples = sum(distribution.values())
        
        # Calculate balanced class weights
        class_weights = {}
        for class_id, count in distribution.items():
            weight = total_samples / (len(distribution) * count)
            class_weights[class_id - 1] = weight  # Convert to 0-based indexing
        
        self.logger.info(f"Class weights: {class_weights}")
        return class_weights
    
    def save_preprocessor_state(self, filepath: str = None):
        """
        Save feature scaler state.
        
        Args:
            filepath: Path to save scaler
        """
        if self.feature_scaler is None:
            raise ValueError("Feature scaler not fitted")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"{self.paths['metadata']}/feature_scaler_{timestamp}.pkl"
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        self.logger.info(f"Feature scaler saved: {filepath}")
        return filepath
    
    def load_preprocessor_state(self, filepath: str):
        """
        Load feature scaler state.
        
        Args:
            filepath: Path to load scaler from
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        self.logger.info(f"Feature scaler loaded: {filepath}")

def main():
    """Test the daily data loader."""
    print("Testing Daily Data Loader...")
    
    try:
        # Initialize data loader
        data_loader = DailyDataLoader()
        
        # Load dataset index
        print("📊 Loading dataset index...")
        samples = data_loader.load_dataset_index("test_daily_nq")
        print(f"   Loaded {len(samples)} samples")
        
        # Split dataset
        print("🔀 Splitting dataset...")
        train_samples, val_samples, test_samples = data_loader.split_dataset()
        
        # Fit feature scaler on training data
        print("⚖️  Fitting feature scaler...")
        data_loader.fit_feature_scaler(train_samples)
        
        # Create datasets
        print("🏗️  Creating TensorFlow datasets...")
        train_dataset = data_loader.get_train_dataset(batch_size=4)
        val_dataset = data_loader.get_val_dataset(batch_size=4)
        test_dataset = data_loader.get_test_dataset(batch_size=4)
        
        # Test data loading
        print("🧪 Testing data loading...")
        for batch in train_dataset.take(1):
            images, features, labels = batch
            print(f"   Batch shapes: Images {images.shape}, Features {features.shape}, Labels {labels.shape}")
            print(f"   Sample label: {labels[0].numpy()}")
            print(f"   Sample features: {features[0].numpy()[:3]}...")  # First 3 features
        
        # Show class distribution
        print("📈 Class distribution:")
        train_dist = data_loader.get_class_distribution(train_samples)
        for pattern, count in train_dist.items():
            label = data_loader.config['classification']['labels'][pattern]
            percentage = (count / len(train_samples)) * 100
            print(f"   {pattern}: {label} - {count} ({percentage:.1f}%)")
        
        # Calculate class weights
        print("⚖️  Class weights:")
        class_weights = data_loader.get_class_weights()
        for class_id, weight in class_weights.items():
            print(f"   Class {class_id}: {weight:.3f}")
        
        print("✅ Data loader test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())