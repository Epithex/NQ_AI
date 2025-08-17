#!/usr/bin/env python3
"""
Pure Visual Data Loader for NQ Pattern Analysis
Loads chart images only (no numerical features)
"""

import tensorflow as tf
import json
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.model_selection import train_test_split
from collections import Counter


class VisualDataLoader:
    """Data loader for pure visual chart analysis."""

    def __init__(
        self,
        data_root: str = "data",
        image_size: int = 224,
        batch_size: int = 16,
        validation_split: float = 0.2,
        test_split: float = 0.2,
        random_state: int = 42,
    ):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_state = random_state

        # Load and prepare data
        self.samples = self._load_samples()
        self.train_samples, self.val_samples, self.test_samples = self._split_data()
        self.class_weights = self._calculate_class_weights()

        logging.info(f"Visual data loader initialized:")
        logging.info(f"  - Total samples: {len(self.samples)}")
        logging.info(f"  - Train: {len(self.train_samples)}")
        logging.info(f"  - Validation: {len(self.val_samples)}")
        logging.info(f"  - Test: {len(self.test_samples)}")
        logging.info(f"  - Image size: {image_size}x{image_size}")
        logging.info(f"  - Batch size: {batch_size}")

    def _load_samples(self) -> List[Dict]:
        """Load all samples with image paths and labels."""
        samples = []

        for label_file in self.labels_dir.glob("daily_sample_*.json"):
            with open(label_file, "r") as f:
                data = json.load(f)

            # Extract image path and label
            image_file = (
                self.images_dir / f"daily_chart_{data['date'].replace('-', '')}.png"
            )

            if image_file.exists():
                sample = {
                    "image_path": str(image_file),
                    "label": data["pattern_rank"] - 1,  # Convert to 0-based indexing
                    "date": data["date"],
                    "pattern_name": data["pattern_label"],
                }
                samples.append(sample)

        logging.info(f"Loaded {len(samples)} valid samples")
        return samples

    def _split_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets."""
        # Extract labels for stratified splitting
        labels = [sample["label"] for sample in self.samples]

        # First split: separate test set
        train_val_samples, test_samples = train_test_split(
            self.samples,
            test_size=self.test_split,
            stratify=labels,
            random_state=self.random_state,
        )

        # Second split: separate train and validation
        train_val_labels = [sample["label"] for sample in train_val_samples]
        val_size = self.validation_split / (1 - self.test_split)

        train_samples, val_samples = train_test_split(
            train_val_samples,
            test_size=val_size,
            stratify=train_val_labels,
            random_state=self.random_state,
        )

        return train_samples, val_samples, test_samples

    def _calculate_class_weights(self) -> Dict[int, float]:
        """Calculate class weights for balanced training."""
        train_labels = [sample["label"] for sample in self.train_samples]
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        num_classes = len(class_counts)

        # Calculate weights using inverse frequency
        class_weights = {}
        for class_id, count in class_counts.items():
            class_weights[class_id] = total_samples / (num_classes * count)

        logging.info("Class distribution and weights:")
        for class_id in sorted(class_weights.keys()):
            count = class_counts[class_id]
            weight = class_weights[class_id]
            logging.info(f"  Class {class_id}: {count} samples, weight: {weight:.3f}")

        return class_weights

    def _preprocess_image(self, image_path: str) -> tf.Tensor:
        """Load and preprocess a single image."""
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)

        # Resize to target size
        image = tf.image.resize(image, [self.image_size, self.image_size])

        # Normalize to [0, 1] range (ViT expects this)
        image = image / 255.0

        # Apply ImageNet normalization (ViT pre-training)
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std

        return image

    def _create_dataset_from_samples(
        self, samples: List[Dict], shuffle: bool = True
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset from samples."""
        image_paths = [sample["image_path"] for sample in samples]
        labels = [sample["label"] for sample in samples]

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            {"image_path": image_paths, "label": labels}
        )

        # Preprocess images
        def preprocess_fn(data):
            image = self._preprocess_image(data["image_path"])
            label = data["label"]
            return image, label

        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=self.random_state)

        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_train_dataset(self) -> tf.data.Dataset:
        """Get training dataset."""
        return self._create_dataset_from_samples(self.train_samples, shuffle=True)

    def get_val_dataset(self) -> tf.data.Dataset:
        """Get validation dataset."""
        return self._create_dataset_from_samples(self.val_samples, shuffle=False)

    def get_test_dataset(self) -> tf.data.Dataset:
        """Get test dataset."""
        return self._create_dataset_from_samples(self.test_samples, shuffle=False)

    def get_class_weights(self) -> Dict[int, float]:
        """Get class weights for balanced training."""
        return self.class_weights

    def get_dataset_info(self) -> Dict:
        """Get dataset information."""
        return {
            "total_samples": len(self.samples),
            "train_samples": len(self.train_samples),
            "val_samples": len(self.val_samples),
            "test_samples": len(self.test_samples),
            "num_classes": 4,
            "class_weights": self.class_weights,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
        }


if __name__ == "__main__":
    # Test data loader
    logging.basicConfig(level=logging.INFO)

    print("Creating visual data loader...")
    loader = VisualDataLoader()

    print("\nDataset info:")
    info = loader.get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nTesting data loading...")
    train_ds = loader.get_train_dataset()

    # Test one batch
    for images, labels in train_ds.take(1):
        print(f"Batch shape: images={images.shape}, labels={labels.shape}")
        print(
            f"Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]"
        )
        print(f"Labels: {labels.numpy()}")
        break

    print("Visual data loader test complete!")
