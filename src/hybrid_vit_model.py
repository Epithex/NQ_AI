#!/usr/bin/env python3
"""
Hybrid Vision Transformer Model for NQ Daily Pattern Classification
Combines chart images with numerical features for enhanced pattern recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Tuple, List, Optional
import yaml
import logging
import os
from datetime import datetime


class HybridViTModel:
    """Hybrid Vision Transformer model combining images and numerical features."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the hybrid ViT model."""
        self.config = self.load_config(config_path)
        self.model_config = self.config["model"]
        self.feature_config = self.config["features"]
        self.setup_logging()

        # Model components
        self.image_model = None
        self.feature_model = None
        self.hybrid_model = None

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging for the model."""
        log_dir = self.config["paths"]["logs_dir"]
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(f"{log_dir}/hybrid_vit.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def create_image_encoder(self) -> keras.Model:
        """
        Create Vision Transformer encoder for chart images.

        Returns:
            Image encoder model
        """
        self.logger.info("Creating ViT image encoder...")

        # Image input
        image_input = layers.Input(
            shape=(self.model_config["image_size"], self.model_config["image_size"], 3),
            name="chart_image",
        )

        # Data augmentation
        augmented = self.add_data_augmentation(image_input)

        # Patch embedding
        patches = self.create_patches(augmented)
        encoded_patches = self.encode_patches(patches)

        # Add positional encoding
        patch_embeddings = self.add_positional_encoding(encoded_patches)

        # Transformer blocks
        for i in range(self.model_config["num_transformer_blocks"]):
            patch_embeddings = self.transformer_block(
                patch_embeddings, f"transformer_block_{i}"
            )

        # Global average pooling
        representation = layers.GlobalAveragePooling1D()(patch_embeddings)

        # Dense layers
        representation = layers.Dense(
            self.model_config["dense_units"], activation="relu", name="image_dense_1"
        )(representation)
        representation = layers.Dropout(self.model_config["dropout_rate"])(
            representation
        )

        representation = layers.Dense(
            self.model_config["dense_units"] // 2,
            activation="relu",
            name="image_dense_2",
        )(representation)

        # Create model
        image_encoder = keras.Model(
            inputs=image_input, outputs=representation, name="image_encoder"
        )

        self.logger.info(
            f"Image encoder created with {image_encoder.count_params()} parameters"
        )
        return image_encoder

    def add_data_augmentation(self, x):
        """Add data augmentation layers."""
        # Use Keras preprocessing layers for augmentation
        augmentation = keras.Sequential(
            [layers.RandomContrast(factor=0.1), layers.RandomBrightness(factor=0.05)]
        )

        return augmentation(x)

    def create_patches(self, x):
        """Create patches from input image using a custom layer."""

        class PatchExtraction(layers.Layer):
            def __init__(self, patch_size, **kwargs):
                super().__init__(**kwargs)
                self.patch_size = patch_size

            def call(self, x):
                # Extract patches
                patches = tf.image.extract_patches(
                    images=x,
                    sizes=[1, self.patch_size, self.patch_size, 1],
                    strides=[1, self.patch_size, self.patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID",
                )

                # Reshape patches
                batch_size = tf.shape(patches)[0]
                num_patches = (224 // self.patch_size) ** 2
                patch_dims = self.patch_size * self.patch_size * 3

                patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
                return patches

        patch_layer = PatchExtraction(self.model_config["patch_size"])
        return patch_layer(x)

    def encode_patches(self, patches):
        """Encode patches with linear projection."""
        projection_dim = self.model_config["projection_dim"]

        encoded = layers.Dense(projection_dim, name="patch_projection")(patches)
        return encoded

    def add_positional_encoding(self, encoded_patches):
        """Add learnable positional encoding."""
        num_patches = (
            self.model_config["image_size"] // self.model_config["patch_size"]
        ) ** 2
        projection_dim = self.model_config["projection_dim"]

        # Learnable position embeddings
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim, name="position_embedding"
        )(positions)

        # Add position embeddings to patch embeddings
        encoded_patches = encoded_patches + position_embedding

        return encoded_patches

    def transformer_block(self, x, name_prefix):
        """Create a transformer block."""
        projection_dim = self.model_config["projection_dim"]
        num_heads = self.model_config["num_heads"]

        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x)

        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            name=f"{name_prefix}_attention",
        )(x1, x1)

        # Skip connection 1
        x2 = layers.Add(name=f"{name_prefix}_add1")([attention_output, x])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x2)

        # MLP
        x3 = layers.Dense(
            projection_dim * 2, activation="gelu", name=f"{name_prefix}_mlp1"
        )(x3)
        x3 = layers.Dropout(self.model_config["dropout_rate"])(x3)
        x3 = layers.Dense(projection_dim, name=f"{name_prefix}_mlp2")(x3)
        x3 = layers.Dropout(self.model_config["dropout_rate"])(x3)

        # Skip connection 2
        x = layers.Add(name=f"{name_prefix}_add2")([x3, x2])

        return x

    def create_feature_encoder(self) -> keras.Model:
        """
        Create encoder for numerical features.

        Returns:
            Feature encoder model
        """
        self.logger.info("Creating numerical feature encoder...")

        num_features = len(self.feature_config["feature_names"])

        # Numerical features input
        feature_input = layers.Input(shape=(num_features,), name="numerical_features")

        # Feature normalization
        normalized_features = layers.BatchNormalization(name="feature_normalization")(
            feature_input
        )

        # Dense layers for feature processing
        x = layers.Dense(
            self.model_config["feature_dense_units"],
            activation="relu",
            name="feature_dense_1",
        )(normalized_features)
        x = layers.Dropout(self.model_config["dropout_rate"])(x)

        x = layers.Dense(
            self.model_config["feature_dense_units"] // 2,
            activation="relu",
            name="feature_dense_2",
        )(x)
        x = layers.Dropout(self.model_config["dropout_rate"])(x)

        feature_representation = layers.Dense(
            self.model_config["feature_output_dim"],
            activation="relu",
            name="feature_representation",
        )(x)

        # Create model
        feature_encoder = keras.Model(
            inputs=feature_input, outputs=feature_representation, name="feature_encoder"
        )

        self.logger.info(
            f"Feature encoder created with {feature_encoder.count_params()} parameters"
        )
        return feature_encoder

    def create_hybrid_model(self) -> keras.Model:
        """
        Create complete hybrid model combining image and feature encoders.

        Returns:
            Complete hybrid model
        """
        self.logger.info("Creating hybrid ViT model...")

        # Create encoders
        self.image_model = self.create_image_encoder()
        self.feature_model = self.create_feature_encoder()

        # Inputs
        image_input = self.image_model.input
        feature_input = self.feature_model.input

        # Get encoded representations
        image_features = self.image_model.output
        numerical_features = self.feature_model.output

        # Combine features
        combined_features = layers.Concatenate(name="feature_fusion")(
            [image_features, numerical_features]
        )

        # Additional fusion layers
        x = layers.Dense(
            self.model_config["fusion_units"], activation="relu", name="fusion_dense_1"
        )(combined_features)
        x = layers.Dropout(self.model_config["dropout_rate"])(x)

        x = layers.Dense(
            self.model_config["fusion_units"] // 2,
            activation="relu",
            name="fusion_dense_2",
        )(x)
        x = layers.Dropout(self.model_config["dropout_rate"])(x)

        # Classification head
        predictions = layers.Dense(
            self.config["classification"]["num_classes"],
            activation="softmax",
            name="pattern_predictions",
        )(x)

        # Create complete model
        self.hybrid_model = keras.Model(
            inputs=[image_input, feature_input],
            outputs=predictions,
            name="hybrid_vit_model",
        )

        total_params = self.hybrid_model.count_params()
        self.logger.info(f"Hybrid model created with {total_params:,} total parameters")

        return self.hybrid_model

    def compile_model(self, learning_rate: float = None):
        """
        Compile the hybrid model with optimizer and loss.

        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.hybrid_model is None:
            raise ValueError("Model must be created before compilation")

        if learning_rate is None:
            learning_rate = self.model_config["learning_rate"]

        # Create optimizer with learning rate schedule
        initial_learning_rate = learning_rate
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=self.model_config["decay_steps"],
            alpha=0.1,
        )

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=self.model_config["weight_decay"]
        )

        # Compile model
        self.hybrid_model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
            ],
        )

        self.logger.info("Model compiled successfully")
        self.logger.info(f"Learning rate: {initial_learning_rate}")
        self.logger.info(f"Weight decay: {self.model_config['weight_decay']}")

    def get_model_summary(self) -> str:
        """
        Get detailed model summary.

        Returns:
            Model summary string
        """
        if self.hybrid_model is None:
            return "Model not created yet"

        # Capture summary
        summary_lines = []
        self.hybrid_model.summary(print_fn=lambda x: summary_lines.append(x))

        return "\n".join(summary_lines)

    def save_model_architecture(self, filepath: str = None):
        """
        Save model architecture to file.

        Args:
            filepath: Path to save architecture visualization
        """
        if self.hybrid_model is None:
            raise ValueError("Model must be created before saving")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.config['paths']['metadata']}/hybrid_vit_architecture_{timestamp}.png"

        # Create visualization
        keras.utils.plot_model(
            self.hybrid_model,
            to_file=filepath,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=150,
        )

        self.logger.info(f"Model architecture saved: {filepath}")
        return filepath

    def load_model_weights(self, weights_path: str):
        """
        Load pre-trained weights.

        Args:
            weights_path: Path to weights file
        """
        if self.hybrid_model is None:
            raise ValueError("Model must be created before loading weights")

        self.hybrid_model.load_weights(weights_path)
        self.logger.info(f"Loaded weights from: {weights_path}")

    def save_model_weights(self, weights_path: str = None):
        """
        Save model weights.

        Args:
            weights_path: Path to save weights
        """
        if self.hybrid_model is None:
            raise ValueError("Model must be created before saving weights")

        if weights_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_path = (
                f"{self.config['paths']['models']}/hybrid_vit_weights_{timestamp}.h5"
            )

        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self.hybrid_model.save_weights(weights_path)
        self.logger.info(f"Weights saved: {weights_path}")
        return weights_path


def main():
    """Test the hybrid ViT model."""
    print("Testing Hybrid ViT Model...")

    try:
        # Initialize model
        model_builder = HybridViTModel()

        # Create model
        print("🏗️  Creating hybrid model...")
        hybrid_model = model_builder.create_hybrid_model()

        # Compile model
        print("⚙️  Compiling model...")
        model_builder.compile_model()

        # Print model summary
        print("📊 Model Summary:")
        print(model_builder.get_model_summary())

        # Test with dummy data
        print("🧪 Testing with dummy data...")

        # Create dummy inputs
        batch_size = 4
        image_size = model_builder.model_config["image_size"]
        num_features = len(model_builder.feature_config["feature_names"])

        dummy_images = np.random.rand(batch_size, image_size, image_size, 3)
        dummy_features = np.random.rand(batch_size, num_features)

        # Test prediction
        predictions = hybrid_model.predict([dummy_images, dummy_features])

        print(f"✅ Model test successful!")
        print(
            f"   Input shapes: Images {dummy_images.shape}, Features {dummy_features.shape}"
        )
        print(f"   Output shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[0]}")

        # Save architecture visualization
        print("💾 Saving model architecture...")
        arch_path = model_builder.save_model_architecture()
        print(f"   Saved to: {arch_path}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
