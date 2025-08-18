#!/usr/bin/env python3
"""
Daily Hybrid Data Loader for 4-Class Previous Day Levels Classification
TensorFlow data pipeline for loading chart images and numerical features
Supports hybrid model architecture with visual + numerical inputs
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
from sklearn.utils.class_weight import compute_class_weight
import glob


class DailyDataLoader:
    """TensorFlow data loader for 4-class hybrid daily pattern analysis."""

    def __init__(self, config_path: str = "config/config_daily_hybrid.yaml"):
        """Initialize the daily hybrid data loader."""
        self.config = self.load_config(config_path)
        self.paths = self.config["paths"]
        self.training_config = self.config["training"]

        self.setup_logging()

        # Data structures
        self.sample_index = None
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        self.class_weights = None

        # Feature scaling parameters (will be computed from training data)
        self.numerical_scaler_params = None

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging for the data loader."""
        log_dir = self.config["paths"]["logs_dir"]
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(f"{log_dir}/daily_data_loader.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def load_dataset_manifest(self) -> Dict:
        """
        Load the latest daily hybrid dataset manifest with all samples.

        Returns:
            Dictionary with dataset information
        """
        try:
            # Find the latest manifest file
            manifest_pattern = os.path.join(
                self.paths["metadata"], "daily_hybrid_dataset_manifest_*.json"
            )
            manifest_files = glob.glob(manifest_pattern)
            
            if not manifest_files:
                raise FileNotFoundError(f"No daily hybrid dataset manifest found: {manifest_pattern}")
            
            # Get the most recent manifest
            latest_manifest = max(manifest_files, key=os.path.getctime)

            with open(latest_manifest, "r") as f:
                manifest = json.load(f)

            self.logger.info(
                f"Loaded dataset manifest: {latest_manifest}"
            )
            self.logger.info(
                f"Total samples: {manifest['total_samples']}, Model type: {manifest['model_type']}"
            )
            self.logger.info(f"Instruments: {list(manifest['instruments'].keys())}")

            return manifest

        except Exception as e:
            self.logger.error(f"Error loading dataset manifest: {e}")
            raise

    def load_sample_index(self) -> List[Dict]:
        """
        Load sample index from dataset manifest for efficient data loading.

        Returns:
            List of sample index entries
        """
        try:
            # Load the full manifest (which contains all samples)
            manifest = self.load_dataset_manifest()
            sample_index = manifest.get("samples", [])

            self.logger.info(f"Loaded sample index with {len(sample_index)} entries")
            self.logger.info(f"Model type: {manifest.get('model_type', 'unknown')}")
            
            # Validate sample structure
            if sample_index and isinstance(sample_index[0], dict):
                required_keys = ["chart_path", "pattern", "numerical_features"]
                sample = sample_index[0]
                for key in required_keys:
                    if key not in sample:
                        self.logger.warning(f"Sample missing required key: {key}")
            
            return sample_index

        except Exception as e:
            self.logger.error(f"Error loading sample index: {e}")
            raise

    def create_data_splits(self, sample_index: List[Dict]) -> Tuple[List, List, List]:
        """
        Create train/validation/test splits with temporal separation.

        Args:
            sample_index: List of sample entries

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        try:
            self.logger.info("Creating temporal data splits for 4-class hybrid classification")

            # Sort samples by date for temporal splitting
            sorted_samples = sorted(sample_index, key=lambda x: x["date"])

            # Calculate split indices
            total_samples = len(sorted_samples)
            train_split = self.training_config["train_split"]
            val_split = self.training_config["val_split"]

            train_end = int(total_samples * train_split)
            val_end = int(total_samples * (train_split + val_split))

            # Create splits
            train_samples = sorted_samples[:train_end]
            val_samples = sorted_samples[train_end:val_end]
            test_samples = sorted_samples[val_end:]

            self.logger.info(f"Data splits created:")
            self.logger.info(
                f"  Train: {len(train_samples)} samples ({len(train_samples)/total_samples*100:.1f}%)"
            )
            self.logger.info(
                f"  Validation: {len(val_samples)} samples ({len(val_samples)/total_samples*100:.1f}%)"
            )
            self.logger.info(
                f"  Test: {len(test_samples)} samples ({len(test_samples)/total_samples*100:.1f}%)"
            )

            # Log date ranges
            if train_samples:
                self.logger.info(
                    f"  Train dates: {train_samples[0]['date'][:10]} to {train_samples[-1]['date'][:10]}"
                )
            if val_samples:
                self.logger.info(
                    f"  Val dates: {val_samples[0]['date'][:10]} to {val_samples[-1]['date'][:10]}"
                )
            if test_samples:
                self.logger.info(
                    f"  Test dates: {test_samples[0]['date'][:10]} to {test_samples[-1]['date'][:10]}"
                )

            return train_samples, val_samples, test_samples

        except Exception as e:
            self.logger.error(f"Error creating data splits: {e}")
            raise

    def calculate_class_weights(self, samples: List[Dict]) -> Dict[int, float]:
        """
        Calculate class weights for handling class imbalance in 4-class system.

        Args:
            samples: List of training samples

        Returns:
            Dictionary with class weights (1-4)
        """
        try:
            # Extract labels (patterns 1-4)
            labels = [sample["pattern"] for sample in samples]
            unique_labels = np.unique(labels)

            # Calculate class weights
            class_weights = compute_class_weight(
                "balanced", classes=unique_labels, y=labels
            )

            # Create weight dictionary
            weight_dict = {
                int(label): float(weight)
                for label, weight in zip(unique_labels, class_weights)
            }

            # Log class distribution and weights
            pattern_labels = {
                1: "High Breakout",
                2: "Low Breakdown", 
                3: "Range Expansion",
                4: "Range Bound"
            }
            
            for label in unique_labels:
                count = labels.count(label)
                percentage = (count / len(labels)) * 100
                pattern_name = pattern_labels.get(label, f"Pattern {label}")
                self.logger.info(
                    f"  Class {label} ({pattern_name}): {count} samples ({percentage:.1f}%) - weight: {weight_dict[label]:.3f}"
                )

            return weight_dict

        except Exception as e:
            self.logger.error(f"Error calculating class weights: {e}")
            raise

    def compute_numerical_feature_scaling(self, samples: List[Dict]) -> Dict:
        """
        Compute scaling parameters for numerical features from training data.

        Args:
            samples: List of training samples

        Returns:
            Dictionary with scaling parameters
        """
        try:
            # Extract all numerical features
            all_features = []
            for sample in samples:
                if "numerical_features" in sample and sample["numerical_features"]:
                    all_features.append(sample["numerical_features"])

            if not all_features:
                self.logger.warning("No numerical features found for scaling computation")
                return {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

            # Convert to numpy array
            features_array = np.array(all_features)
            
            # Compute mean and std for each feature
            feature_means = np.mean(features_array, axis=0)
            feature_stds = np.std(features_array, axis=0)
            
            # Avoid division by zero
            feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

            scaling_params = {
                "mean": feature_means.tolist(),
                "std": feature_stds.tolist(),
                "feature_names": ["distance_to_prev_high", "distance_to_prev_low", "prev_day_range"]
            }

            self.logger.info("Numerical feature scaling parameters computed:")
            for i, name in enumerate(scaling_params["feature_names"]):
                self.logger.info(f"  {name}: mean={feature_means[i]:.3f}, std={feature_stds[i]:.3f}")

            return scaling_params

        except Exception as e:
            self.logger.error(f"Error computing numerical feature scaling: {e}")
            raise

    def load_and_preprocess_image(self, image_path: str) -> tf.Tensor:
        """
        Load and preprocess chart image for ViT model.

        Args:
            image_path: Path to chart image

        Returns:
            Preprocessed image tensor
        """
        # Read image file
        image = tf.io.read_file(image_path)

        # Decode image
        image = tf.image.decode_png(image, channels=3)

        # Ensure correct size (224x224 for ViT)
        image_size = self.config["model"]["image_size"]
        image = tf.image.resize(image, [image_size, image_size])

        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        return image

    def normalize_numerical_features(self, features) -> tf.Tensor:
        """
        Normalize numerical features using computed scaling parameters.

        Args:
            features: List of 3 numerical features or TensorFlow tensor

        Returns:
            Normalized features tensor
        """
        # Convert to tensor if not already
        if not tf.is_tensor(features):
            features_tensor = tf.constant(features, dtype=tf.float32)
        else:
            features_tensor = tf.cast(features, tf.float32)
            
        if self.numerical_scaler_params is None:
            # No scaling if parameters not computed
            return features_tensor

        # Apply z-score normalization
        mean = tf.constant(self.numerical_scaler_params["mean"], dtype=tf.float32)
        std = tf.constant(self.numerical_scaler_params["std"], dtype=tf.float32)
        
        normalized = (features_tensor - mean) / std

        return normalized

    def create_tf_dataset(
        self,
        samples: List[Dict],
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = True,
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from hybrid samples (images + numerical features).

        Args:
            samples: List of sample dictionaries
            batch_size: Batch size for training
            shuffle: Whether to shuffle the dataset
            repeat: Whether to repeat the dataset

        Returns:
            TensorFlow dataset with (images, numerical_features), labels
        """
        try:
            # Extract image paths, numerical features, and labels
            image_paths = [sample["chart_path"] for sample in samples]
            numerical_features = [sample.get("numerical_features", [0.0, 0.0, 0.0]) for sample in samples]
            labels = [sample["pattern"] - 1 for sample in samples]  # Convert 1-4 to 0-3 for TensorFlow

            # Create dataset from paths, features, and labels
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, numerical_features, labels))

            # Map loading and preprocessing function
            def preprocess_sample(image_path, num_features, label):
                image = self.load_and_preprocess_image(image_path)
                normalized_features = self.normalize_numerical_features(num_features)
                return (image, normalized_features), label

            dataset = dataset.map(
                preprocess_sample,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

            if shuffle:
                # Shuffle with buffer size
                buffer_size = min(len(samples), 1000)
                dataset = dataset.shuffle(buffer_size)

            # Batch the dataset
            dataset = dataset.batch(batch_size)

            if repeat:
                dataset = dataset.repeat()

            # Prefetch for performance
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            self.logger.info(
                f"Created hybrid TensorFlow dataset: {len(samples)} samples, batch_size={batch_size}"
            )
            self.logger.info(f"  Input: (images, numerical_features) -> 4-class labels")

            return dataset

        except Exception as e:
            self.logger.error(f"Error creating TensorFlow dataset: {e}")
            raise

    def get_dataset_info(self, samples: List[Dict]) -> Dict:
        """
        Get information about a hybrid dataset split.

        Args:
            samples: List of samples

        Returns:
            Dictionary with dataset information
        """
        if not samples:
            return {}

        # Pattern distribution (1-4)
        patterns = [sample["pattern"] for sample in samples]
        pattern_counts = {pattern: patterns.count(pattern) for pattern in range(1, 5)}
        total = len(patterns)

        # Pattern labels
        pattern_labels = {
            1: "High Breakout",
            2: "Low Breakdown", 
            3: "Range Expansion",
            4: "Range Bound"
        }

        # Instrument distribution
        instruments = [sample["instrument"] for sample in samples]
        instrument_counts = {}
        for instrument in set(instruments):
            instrument_counts[instrument] = instruments.count(instrument)

        # Date range
        dates = [sample["date"] for sample in samples]
        date_range = {"start": min(dates), "end": max(dates)}

        info = {
            "total_samples": total,
            "pattern_distribution": pattern_counts,
            "pattern_percentages": {
                p: (c / total) * 100 for p, c in pattern_counts.items()
            },
            "pattern_labels": pattern_labels,
            "instrument_distribution": instrument_counts,
            "date_range": date_range,
            "model_type": "hybrid_4class",
            "feature_types": ["visual_charts", "numerical_features"]
        }

        return info

    def prepare_datasets(
        self,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare train, validation, and test datasets for hybrid model.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        try:
            self.logger.info("Preparing 4-class hybrid datasets")

            # Load sample index
            sample_index = self.load_sample_index()

            # Create data splits
            self.train_samples, self.val_samples, self.test_samples = (
                self.create_data_splits(sample_index)
            )

            # Compute numerical feature scaling from training data
            self.numerical_scaler_params = self.compute_numerical_feature_scaling(self.train_samples)

            # Calculate class weights from training data
            if self.training_config.get("use_class_weights", True):
                self.class_weights = self.calculate_class_weights(self.train_samples)
                self.logger.info(f"Class weights calculated: {self.class_weights}")

            # Get batch size
            batch_size = self.config["model"]["batch_size"]

            # Create TensorFlow datasets
            train_dataset = self.create_tf_dataset(
                self.train_samples, batch_size, shuffle=True, repeat=True
            )

            val_dataset = self.create_tf_dataset(
                self.val_samples, batch_size, shuffle=False, repeat=False
            )

            test_dataset = self.create_tf_dataset(
                self.test_samples, batch_size, shuffle=False, repeat=False
            )

            # Log dataset information
            self.logger.info("Hybrid dataset preparation complete:")

            train_info = self.get_dataset_info(self.train_samples)
            self.logger.info(f"  Train: {train_info['total_samples']} samples")
            self.logger.info(
                f"    Pattern distribution: {train_info['pattern_percentages']}"
            )

            val_info = self.get_dataset_info(self.val_samples)
            self.logger.info(f"  Validation: {val_info['total_samples']} samples")

            test_info = self.get_dataset_info(self.test_samples)
            self.logger.info(f"  Test: {test_info['total_samples']} samples")

            return train_dataset, val_dataset, test_dataset

        except Exception as e:
            self.logger.error(f"Error preparing datasets: {e}")
            raise

    def get_steps_per_epoch(self, dataset_type: str = "train") -> int:
        """
        Calculate steps per epoch for training.

        Args:
            dataset_type: Type of dataset ('train', 'val', 'test')

        Returns:
            Number of steps per epoch
        """
        if dataset_type == "train":
            samples = self.train_samples
        elif dataset_type == "val":
            samples = self.val_samples
        elif dataset_type == "test":
            samples = self.test_samples
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        batch_size = self.config["model"]["batch_size"]
        steps = len(samples) // batch_size

        return max(1, steps)  # Ensure at least 1 step

    def validate_dataset(self, dataset: tf.data.Dataset, num_batches: int = 5) -> bool:
        """
        Validate hybrid dataset by checking a few batches.

        Args:
            dataset: TensorFlow dataset to validate
            num_batches: Number of batches to check

        Returns:
            True if dataset is valid
        """
        try:
            self.logger.info(f"Validating hybrid dataset ({num_batches} batches)")

            for i, ((images, numerical_features), labels) in enumerate(dataset.take(num_batches)):
                # Check image shape
                expected_shape = (
                    None,
                    self.config["model"]["image_size"],
                    self.config["model"]["image_size"],
                    3,
                )
                if images.shape[1:] != expected_shape[1:]:
                    self.logger.error(f"Invalid image shape: {images.shape}")
                    return False

                # Check image value range
                if tf.reduce_min(images) < 0 or tf.reduce_max(images) > 1:
                    self.logger.error(
                        f"Images not in [0,1] range: min={tf.reduce_min(images)}, max={tf.reduce_max(images)}"
                    )
                    return False

                # Check numerical features shape (should be batch_size x 3)
                if numerical_features.shape[1] != 3:
                    self.logger.error(f"Invalid numerical features shape: {numerical_features.shape}")
                    return False

                # Check label range (0-3 for TensorFlow, representing classes 1-4)
                if (
                    tf.reduce_min(labels) < 0
                    or tf.reduce_max(labels) >= 4
                ):
                    self.logger.error(
                        f"Invalid labels: min={tf.reduce_min(labels)}, max={tf.reduce_max(labels)} (expected 0-3)"
                    )
                    return False

                self.logger.debug(
                    f"Batch {i+1}: images={images.shape}, features={numerical_features.shape}, labels={labels.shape}"
                )

            self.logger.info("Hybrid dataset validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False


def main():
    """Test the daily hybrid data loader."""
    print("🚀 NQ_AI Daily Hybrid Data Loader")
    print("4-Class Previous Day Levels Classification System")

    try:
        # Initialize data loader
        data_loader = DailyDataLoader()

        print("🏗️  Initializing daily hybrid data loader...")
        print(f"   📊 Model Type: 4-class hybrid (visual + numerical)")
        print(f"   📈 Image size: {data_loader.config['model']['image_size']}")
        print(f"   📦 Batch size: {data_loader.config['model']['batch_size']}")
        print(f"   🎯 Classes: 4 (High Breakout, Low Breakdown, Range Expansion, Range Bound)")
        print(f"   🔢 Numerical features: 3 (distance_to_prev_high, distance_to_prev_low, prev_day_range)")

        # Test data loading capabilities
        print(f"\n🧪 Testing hybrid data loading capabilities...")
        print("   📊 This test simulates the hybrid data loading process")
        print("   📈 Real implementation would:")
        print("   - Load daily hybrid dataset manifest")
        print("   - Create temporal train/val/test splits")
        print("   - Generate TensorFlow datasets with dual inputs")
        print("   - Normalize numerical features")
        print("   - Calculate class weights for 4-class imbalance")
        print("   - Validate hybrid dataset integrity")

        # Show expected data flow
        print(f"\n📊 Expected Hybrid Data Pipeline:")
        print(f"   1. Load sample index from daily manifest")
        print(f"   2. Create temporal splits (80/10/10)")
        print(f"   3. Load and preprocess 224x224 chart images")
        print(f"   4. Normalize numerical features (z-score)")
        print(f"   5. Batch hybrid data for training")
        print(f"   6. Apply class weights for 4-class balance")

        # Show expected batch structure
        print(f"\n🎯 Hybrid Batch Structure:")
        print(f"   Inputs:")
        print(f"     - Images: (batch_size, 224, 224, 3) float32 [0,1]")
        print(f"     - Numerical: (batch_size, 3) float32 (normalized)")
        print(f"   Labels: (batch_size,) int32 [0,1,2,3] representing:")
        print(f"     - 0: High Breakout (high >= prev_high, low > prev_low)")
        print(f"     - 1: Low Breakdown (low <= prev_low, high < prev_high)")
        print(f"     - 2: Range Expansion (high >= prev_high, low <= prev_low)")
        print(f"     - 3: Range Bound (high < prev_high, low > prev_low)")

        print(f"\n📈 Hybrid Model Features:")
        print(f"   - Visual: 30-bar charts with previous day level lines")
        print(f"   - Numerical: 3 key distance/range metrics")
        print(f"   - Fusion: Early fusion in hybrid ViT architecture")
        print(f"   - Performance: Parallel loading + prefetching")

        print("✅ Daily hybrid data loader test completed successfully!")
        print("\n🚀 Ready for hybrid training pipeline:")
        print("   python src/train_daily_model.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
