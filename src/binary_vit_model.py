#!/usr/bin/env python3
"""
Binary Visual ViT-Base Model for Bullish/Bearish Pattern Classification
87M parameter Google ViT-Base-Patch16-224 optimized for binary visual learning
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


class BinaryViTModel:
    """Binary visual ViT-Base model for bullish/bearish chart pattern classification."""

    def __init__(self, config_path: str = "config/config_binary_visual.yaml"):
        """Initialize the binary visual ViT-Base model."""
        self.config = self.load_config(config_path)
        self.model_config = self.config["model"]
        self.setup_logging()

        # Model components
        self.model = None

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
                logging.FileHandler(f"{log_dir}/binary_vit.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def create_patch_embedding(self, input_layer):
        """Create patch embedding layer for ViT."""
        patch_size = self.model_config["patch_size"]
        hidden_size = self.model_config["hidden_size"]
        image_size = self.model_config["image_size"]

        # Extract patches
        patches = layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="patch_embedding",
        )(input_layer)

        # Calculate number of patches
        num_patches = (image_size // patch_size) ** 2

        # Reshape to sequence format
        patches = layers.Reshape((num_patches, hidden_size))(patches)

        return patches

    def transformer_block(self, x, name_prefix):
        """Create a ViT-Base transformer block."""
        hidden_size = self.model_config["hidden_size"]
        num_heads = self.model_config["num_heads"]
        mlp_dim = self.model_config["mlp_dim"]
        dropout_rate = self.model_config["dropout_rate"]

        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x)

        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dropout=dropout_rate,
            name=f"{name_prefix}_attention",
        )(x1, x1)

        # Skip connection 1
        x2 = layers.Add(name=f"{name_prefix}_add1")([attention_output, x])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x2)

        # MLP (Feed Forward Network)
        x3 = layers.Dense(mlp_dim, activation="gelu", name=f"{name_prefix}_mlp1")(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        x3 = layers.Dense(hidden_size, name=f"{name_prefix}_mlp2")(x3)
        x3 = layers.Dropout(dropout_rate)(x3)

        # Skip connection 2
        output = layers.Add(name=f"{name_prefix}_add2")([x3, x2])

        return output

    def create_model(self) -> keras.Model:
        """
        Create binary visual ViT-Base model.

        Returns:
            Complete binary visual model
        """
        self.logger.info("Creating binary visual ViT-Base model...")

        image_size = self.model_config["image_size"]
        hidden_size = self.model_config["hidden_size"]
        num_layers = self.model_config["num_layers"]
        num_classes = self.config["classification"]["num_classes"]  # 3 classes
        patch_size = self.model_config["patch_size"]
        classification_head_size = self.model_config["classification_head_size"]

        # Input layer (pure visual - no numerical features)
        image_input = layers.Input(
            shape=(image_size, image_size, 3), name="chart_image"
        )

        # Patch embedding
        patches = self.create_patch_embedding(image_input)

        # Positional encoding with class token
        class PatchPositionEmbedding(layers.Layer):
            def __init__(self, num_patches, hidden_size, **kwargs):
                super().__init__(**kwargs)
                self.num_patches = num_patches
                self.hidden_size = hidden_size

                # Class token
                self.class_token = self.add_weight(
                    shape=(1, 1, hidden_size),
                    initializer="random_normal",
                    trainable=True,
                    name="class_token",
                )

                # Position embedding
                self.position_embedding = layers.Embedding(
                    input_dim=num_patches + 1,
                    output_dim=hidden_size,
                    name="position_embedding",
                )

            def call(self, patches):
                batch_size = tf.shape(patches)[0]

                # Tile class token
                class_tokens = tf.tile(self.class_token, [batch_size, 1, 1])

                # Concatenate class token with patches
                patches_with_cls = tf.concat([class_tokens, patches], axis=1)

                # Add positional encoding
                positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
                encoded = patches_with_cls + self.position_embedding(positions)

                return encoded

        # Apply positional encoding
        num_patches = (image_size // patch_size) ** 2
        encoded_patches = PatchPositionEmbedding(
            num_patches, hidden_size, name="position_encoding"
        )(patches)

        # ViT-Base transformer blocks (12 layers)
        x = encoded_patches
        for i in range(num_layers):
            x = self.transformer_block(x, f"transformer_block_{i}")

        # Final layer normalization
        x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

        # Extract class token representation
        class_token_output = x[:, 0]  # First token is class token

        # Binary classification head
        x = layers.Dense(
            classification_head_size, activation="gelu", name="pre_classification"
        )(class_token_output)
        x = layers.Dropout(self.model_config["dropout_rate"])(x)

        # Additional layer for better binary separation
        x = layers.Dense(
            classification_head_size // 2, activation="gelu", name="binary_features"
        )(x)
        x = layers.Dropout(self.model_config["dropout_rate"] * 0.5)(x)

        # Final classification layer (3 classes: Bearish, Bullish, Neutral)
        predictions = layers.Dense(
            num_classes, activation="softmax", name="binary_classification"
        )(x)

        # Create model
        self.model = keras.Model(
            inputs=image_input, outputs=predictions, name="binary_visual_vit_base"
        )

        total_params = self.model.count_params()
        self.logger.info(
            f"Binary visual ViT-Base model created with {total_params:,} parameters"
        )

        return self.model

    def compile_model(self, learning_rate: float = None):
        """
        Compile the binary visual ViT model.

        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model must be created before compilation")

        if learning_rate is None:
            learning_rate = self.model_config["learning_rate"]

        # Learning rate schedule for binary classification
        if self.config["training"].get("lr_scheduling", True):
            lr_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=1000,  # Will be updated based on dataset size
                alpha=0.1,
            )
        else:
            lr_schedule = learning_rate

        # Optimizer optimized for binary classification
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=self.model_config["weight_decay"]
        )

        # Binary classification metrics
        metrics = [
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(name="f1_score"),
            keras.metrics.AUC(name="auc", multi_label=False),
        ]

        # Loss function for 3-class classification
        loss = "sparse_categorical_crossentropy"

        # Compile model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.logger.info("Binary visual ViT-Base model compiled successfully")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Weight decay: {self.model_config['weight_decay']}")
        self.logger.info(f"Loss function: {loss}")
        self.logger.info(f"Primary metric: {self.config['training']['primary_metric']}")

    def create_class_weights(
        self, class_distribution: Dict[int, int]
    ) -> Dict[int, float]:
        """
        Create class weights for handling class imbalance in binary classification.

        Args:
            class_distribution: Dictionary with class counts

        Returns:
            Dictionary with class weights
        """
        total_samples = sum(class_distribution.values())
        num_classes = len(class_distribution)

        class_weights = {}
        for class_id, count in class_distribution.items():
            # Inverse frequency weighting
            class_weights[class_id] = total_samples / (num_classes * count)

        self.logger.info(f"Class weights calculated: {class_weights}")
        return class_weights

    def get_model_summary(self) -> str:
        """Get detailed model summary."""
        if self.model is None:
            return "Model not created yet"

        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)

    def save_model_architecture(self, filepath: str = None):
        """Save model architecture visualization."""
        if self.model is None:
            raise ValueError("Model must be created before saving")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.config['paths']['metadata']}/binary_vit_architecture_{timestamp}.png"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        keras.utils.plot_model(
            self.model,
            to_file=filepath,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=150,
        )

        self.logger.info(f"Binary model architecture saved: {filepath}")
        return filepath

    def load_weights(self, weights_path: str):
        """Load pre-trained weights."""
        if self.model is None:
            raise ValueError("Model must be created before loading weights")

        self.model.load_weights(weights_path)
        self.logger.info(f"Loaded weights from: {weights_path}")

    def save_weights(self, weights_path: str = None):
        """Save model weights."""
        if self.model is None:
            raise ValueError("Model must be created before saving weights")

        if weights_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_path = (
                f"{self.config['paths']['models']}/binary_vit_weights_{timestamp}.h5"
            )

        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self.model.save_weights(weights_path)
        self.logger.info(f"Binary model weights saved: {weights_path}")
        return weights_path

    def predict_sentiment(self, image_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict market sentiment from chart images.

        Args:
            image_batch: Batch of chart images

        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.model is None:
            raise ValueError("Model must be created and trained before prediction")

        # Get raw predictions
        predictions = self.model.predict(image_batch, verbose=0)

        # Extract class predictions and confidence
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)

        # Convert to sentiment labels
        label_map = self.config["classification"]["labels"]
        sentiment_labels = [label_map[cls] for cls in predicted_classes]

        return {
            "predictions": predictions,
            "predicted_classes": predicted_classes,
            "sentiment_labels": sentiment_labels,
            "confidence_scores": confidence_scores,
            "class_probabilities": {
                "bearish": predictions[:, 0],
                "bullish": predictions[:, 1],
                "neutral": predictions[:, 2],
            },
        }


def main():
    """Test the binary visual ViT model."""
    print("Testing Binary Visual ViT-Base Model...")

    try:
        # Initialize model
        model_builder = BinaryViTModel()

        # Create model
        print("🏗️  Creating binary visual ViT-Base model...")
        model = model_builder.create_model()

        # Compile model
        print("⚙️  Compiling binary model...")
        model_builder.compile_model()

        # Print model summary
        print("📊 Binary Model Summary:")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input.shape}")
        print(f"   Output shape: {model.output.shape}")
        print(f"   Classes: {model_builder.config['classification']['num_classes']}")

        # Test with dummy data
        print("🧪 Testing with dummy data...")

        batch_size = 4
        image_size = model_builder.model_config["image_size"]
        dummy_images = np.random.rand(batch_size, image_size, image_size, 3)

        # Test prediction
        predictions = model.predict(dummy_images, verbose=0)
        sentiment_results = model_builder.predict_sentiment(dummy_images)

        print(f"✅ Binary visual ViT-Base test successful!")
        print(f"   Input shape: {dummy_images.shape}")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[0]}")
        print(f"   Sentiment labels: {sentiment_results['sentiment_labels']}")
        print(f"   Confidence scores: {sentiment_results['confidence_scores']}")
        print(f"   Model size: ~{model.count_params() * 4 / 1e6:.1f} MB")

        print("🎯 Binary Model Features:")
        print("   - Binary bullish/bearish classification")
        print("   - Pure visual input (224x224x3)")
        print("   - No numerical features")
        print("   - 87M parameter ViT-Base architecture")
        print("   - 3-class output (Bearish/Bullish/Neutral)")
        print("   - Optimized for open vs close analysis")
        print("   - Multi-instrument training ready")

        # Test class weights calculation
        print("\n📊 Testing class weights calculation...")
        test_distribution = {0: 4800, 1: 4900, 2: 300}  # Bearish, Bullish, Neutral
        class_weights = model_builder.create_class_weights(test_distribution)
        print(f"   Class weights: {class_weights}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
