#!/usr/bin/env python3
"""
ViT-Base Hybrid Model for NQ Daily Pattern Classification
87M parameter Google ViT-Base-Patch16-224 combined with numerical features
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


class ViTBaseHybridModel:
    """ViT-Base hybrid model combining Google ViT-Base with numerical features."""

    def __init__(self, config_path: str = "config/config_vit_base.yaml"):
        """Initialize the ViT-Base hybrid model."""
        self.config = self.load_config(config_path)
        self.model_config = self.config["model"]
        self.feature_config = self.config["features"]
        self.setup_logging()

        # Model components
        self.vit_base = None
        self.feature_encoder = None
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
                logging.FileHandler(f"{log_dir}/vit_base_hybrid.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def create_vit_base_encoder(self) -> keras.Model:
        """
        Create ViT-Base encoder using Google's architecture.

        Returns:
            ViT-Base encoder model
        """
        self.logger.info("Creating ViT-Base encoder (87M parameters)...")

        # Image input
        image_input = layers.Input(
            shape=(self.model_config["image_size"], self.model_config["image_size"], 3),
            name="chart_image",
        )

        # Data augmentation (light augmentation for charts)
        augmented = self.add_data_augmentation(image_input)

        # Patch embedding
        patches = self.create_patches(augmented)
        encoded_patches = self.encode_patches(patches)

        # Add positional encoding and class token
        patch_embeddings = self.add_positional_encoding(encoded_patches)

        # ViT-Base transformer blocks (12 layers)
        for i in range(self.model_config["num_layers"]):
            patch_embeddings = self.vit_base_transformer_block(
                patch_embeddings, f"vit_base_block_{i}"
            )

        # Layer normalization before pooling
        patch_embeddings = layers.LayerNormalization(
            epsilon=1e-6, name="vit_base_final_ln"
        )(patch_embeddings)

        # Global average pooling
        representation = layers.GlobalAveragePooling1D(name="vit_base_pool")(
            patch_embeddings
        )

        # Dense projection layers
        representation = layers.Dense(
            self.model_config["fusion_units"],
            activation="gelu",
            name="vit_base_dense_1",
        )(representation)
        representation = layers.Dropout(self.model_config["dropout_rate"])(
            representation
        )

        representation = layers.Dense(
            self.model_config["fusion_units"] // 2,
            activation="gelu",
            name="vit_base_dense_2",
        )(representation)

        # Create model
        vit_base_encoder = keras.Model(
            inputs=image_input, outputs=representation, name="vit_base_encoder"
        )

        self.logger.info(
            f"ViT-Base encoder created with {vit_base_encoder.count_params():,} parameters"
        )
        return vit_base_encoder

    def add_data_augmentation(self, x):
        """Add light data augmentation for charts."""
        # Light augmentation suitable for financial charts
        augmentation = keras.Sequential(
            [layers.RandomContrast(factor=0.05), layers.RandomBrightness(factor=0.03)],
            name="chart_augmentation",
        )

        return augmentation(x)

    def create_patches(self, x):
        """Create patches from input image using ViT-Base patch size (16x16)."""

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
                num_patches = (224 // self.patch_size) ** 2  # 14x14 = 196 patches
                patch_dims = self.patch_size * self.patch_size * 3  # 16*16*3 = 768

                patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
                return patches

        patch_layer = PatchExtraction(self.model_config["patch_size"])
        return patch_layer(x)

    def encode_patches(self, patches):
        """Encode patches with linear projection to ViT-Base hidden size (768)."""
        hidden_size = self.model_config["hidden_size"]  # 768 for ViT-Base

        encoded = layers.Dense(hidden_size, name="vit_base_patch_projection")(patches)
        return encoded

    def add_positional_encoding(self, encoded_patches):
        """Add learnable positional encoding and class token."""
        num_patches = (
            self.model_config["image_size"] // self.model_config["patch_size"]
        ) ** 2
        hidden_size = self.model_config["hidden_size"]

        class ClassTokenAndPositionEmbedding(layers.Layer):
            def __init__(self, num_patches, hidden_size, **kwargs):
                super().__init__(**kwargs)
                self.num_patches = num_patches
                self.hidden_size = hidden_size

                # Create class token as trainable weight
                self.class_token = self.add_weight(
                    shape=(1, 1, hidden_size),
                    initializer="random_normal",
                    trainable=True,
                    name="class_token",
                )

                # Position embedding layer
                self.position_embedding = layers.Embedding(
                    input_dim=num_patches + 1,  # +1 for class token
                    output_dim=hidden_size,
                    name="position_embedding",
                )

            def call(self, x):
                batch_size = tf.shape(x)[0]

                # Tile class token for batch
                class_tokens = tf.tile(self.class_token, [batch_size, 1, 1])

                # Concatenate class token with patches
                x = tf.concat([class_tokens, x], axis=1)

                # Create position indices
                positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)

                # Get position embeddings
                pos_embeddings = self.position_embedding(positions)

                # Add position embeddings
                x = x + pos_embeddings

                return x

        # Apply class token and position encoding
        class_pos_layer = ClassTokenAndPositionEmbedding(
            num_patches, hidden_size, name="vit_base_class_pos"
        )

        return class_pos_layer(encoded_patches)

    def vit_base_transformer_block(self, x, name_prefix):
        """Create a ViT-Base transformer block with proper architecture."""
        hidden_size = self.model_config["hidden_size"]  # 768
        num_heads = self.model_config["num_heads"]  # 12
        mlp_dim = self.model_config["mlp_dim"]  # 3072

        # Layer normalization 1 (pre-norm architecture)
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x)

        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dropout=self.model_config["dropout_rate"],
            name=f"{name_prefix}_attention",
        )(x1, x1)

        # Skip connection 1
        x2 = layers.Add(name=f"{name_prefix}_add1")([attention_output, x])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x2)

        # MLP (Feed Forward Network)
        x3 = layers.Dense(mlp_dim, activation="gelu", name=f"{name_prefix}_mlp1")(x3)
        x3 = layers.Dropout(self.model_config["dropout_rate"])(x3)
        x3 = layers.Dense(hidden_size, name=f"{name_prefix}_mlp2")(x3)
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
            activation="gelu",
            name="feature_dense_1",
        )(normalized_features)
        x = layers.Dropout(self.model_config["dropout_rate"])(x)

        x = layers.Dense(
            self.model_config["feature_dense_units"] // 2,
            activation="gelu",
            name="feature_dense_2",
        )(x)
        x = layers.Dropout(self.model_config["dropout_rate"])(x)

        feature_representation = layers.Dense(
            self.model_config["feature_output_dim"],
            activation="gelu",
            name="feature_representation",
        )(x)

        # Create model
        feature_encoder = keras.Model(
            inputs=feature_input, outputs=feature_representation, name="feature_encoder"
        )

        self.logger.info(
            f"Feature encoder created with {feature_encoder.count_params():,} parameters"
        )
        return feature_encoder

    def create_hybrid_model(self) -> keras.Model:
        """
        Create complete ViT-Base hybrid model combining image and feature encoders.

        Returns:
            Complete hybrid model
        """
        self.logger.info("Creating ViT-Base hybrid model...")

        # Create encoders
        self.vit_base = self.create_vit_base_encoder()
        self.feature_encoder = self.create_feature_encoder()

        # Inputs
        image_input = self.vit_base.input
        feature_input = self.feature_encoder.input

        # Get encoded representations
        image_features = self.vit_base.output
        numerical_features = self.feature_encoder.output

        # Combine features
        combined_features = layers.Concatenate(name="feature_fusion")(
            [image_features, numerical_features]
        )

        # Additional fusion layers
        x = layers.Dense(
            self.model_config["fusion_units"], activation="gelu", name="fusion_dense_1"
        )(combined_features)
        x = layers.Dropout(self.model_config["fusion_dropout"])(x)

        x = layers.Dense(
            self.model_config["fusion_units"] // 2,
            activation="gelu",
            name="fusion_dense_2",
        )(x)
        x = layers.Dropout(self.model_config["fusion_dropout"])(x)

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
            name="vit_base_hybrid_model",
        )

        total_params = self.hybrid_model.count_params()
        self.logger.info(
            f"ViT-Base hybrid model created with {total_params:,} total parameters"
        )

        return self.hybrid_model

    def compile_model(self, learning_rate: float = None):
        """
        Compile the ViT-Base hybrid model with optimizer and loss.

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
                keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.F1Score(name="f1_score"),
            ],
        )

        self.logger.info("ViT-Base hybrid model compiled successfully")
        self.logger.info(f"Learning rate: {initial_learning_rate}")
        self.logger.info(f"Weight decay: {self.model_config['weight_decay']}")

    def get_model_summary(self) -> str:
        """Get detailed model summary."""
        if self.hybrid_model is None:
            return "Model not created yet"

        # Capture summary
        summary_lines = []
        self.hybrid_model.summary(print_fn=lambda x: summary_lines.append(x))

        return "\n".join(summary_lines)

    def save_model_architecture(self, filepath: str = None):
        """Save model architecture visualization."""
        if self.hybrid_model is None:
            raise ValueError("Model must be created before saving")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.config['paths']['metadata']}/vit_base_hybrid_architecture_{timestamp}.png"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

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
        """Load pre-trained weights."""
        if self.hybrid_model is None:
            raise ValueError("Model must be created before loading weights")

        self.hybrid_model.load_weights(weights_path)
        self.logger.info(f"Loaded weights from: {weights_path}")

    def save_model_weights(self, weights_path: str = None):
        """Save model weights."""
        if self.hybrid_model is None:
            raise ValueError("Model must be created before saving weights")

        if weights_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_path = f"{self.config['paths']['models']}/vit_base_hybrid_weights_{timestamp}.h5"

        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self.hybrid_model.save_weights(weights_path)
        self.logger.info(f"Weights saved: {weights_path}")
        return weights_path


def main():
    """Test the ViT-Base hybrid model."""
    print("Testing ViT-Base Hybrid Model...")

    try:
        # Initialize model
        model_builder = ViTBaseHybridModel()

        # Create model
        print("🏗️  Creating ViT-Base hybrid model...")
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
        batch_size = 2  # Small batch for testing
        image_size = model_builder.model_config["image_size"]
        num_features = len(model_builder.feature_config["feature_names"])

        dummy_images = np.random.rand(batch_size, image_size, image_size, 3)
        dummy_features = np.random.rand(batch_size, num_features)

        # Test prediction
        predictions = hybrid_model.predict([dummy_images, dummy_features])

        print(f"✅ ViT-Base hybrid model test successful!")
        print(
            f"   Input shapes: Images {dummy_images.shape}, Features {dummy_features.shape}"
        )
        print(f"   Output shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[0]}")
        print(f"   Total parameters: {hybrid_model.count_params():,}")

        # Save architecture visualization
        print("💾 Saving model architecture...")
        arch_path = model_builder.save_model_architecture()
        print(f"   Saved to: {arch_path}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
