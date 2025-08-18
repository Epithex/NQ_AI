#!/usr/bin/env python3
"""
Hybrid ViT Model for 4-Class Previous Day Levels Classification
Combines visual chart analysis with numerical features for enhanced pattern recognition
Supports both visual ViT-Base backbone and numerical feature fusion
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
    """Hybrid ViT model for 4-class daily pattern analysis with feature fusion."""

    def __init__(self, config_path: str = "config/config_daily_hybrid.yaml"):
        """Initialize the hybrid ViT model."""
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
                logging.FileHandler(f"{log_dir}/hybrid_vit_model.log"),
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
        Create hybrid ViT model with visual and numerical feature fusion.

        Returns:
            Complete hybrid model for 4-class classification
        """
        self.logger.info("Creating hybrid ViT model for 4-class classification...")

        image_size = self.model_config["image_size"]
        hidden_size = self.model_config["hidden_size"]
        num_layers = self.model_config["num_layers"]
        num_classes = 4  # 4-class classification
        patch_size = self.model_config["patch_size"]
        classification_head_size = self.model_config["classification_head_size"]

        # Input layers
        image_input = layers.Input(
            shape=(image_size, image_size, 3), name="chart_image"
        )
        numerical_input = layers.Input(shape=(3,), name="numerical_features")  # 3 numerical features

        # === VISUAL BRANCH (ViT-Base) ===
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
        visual_features = encoded_patches
        for i in range(num_layers):
            visual_features = self.transformer_block(visual_features, f"transformer_block_{i}")

        # Final layer normalization
        visual_features = layers.LayerNormalization(epsilon=1e-6, name="visual_final_ln")(visual_features)

        # Extract class token representation
        visual_class_token = visual_features[:, 0]  # First token is class token

        # Visual feature processing
        visual_processed = layers.Dense(
            classification_head_size, activation="gelu", name="visual_dense_1"
        )(visual_class_token)
        visual_processed = layers.Dropout(self.model_config["dropout_rate"])(visual_processed)

        visual_processed = layers.Dense(
            classification_head_size // 2, activation="gelu", name="visual_dense_2"
        )(visual_processed)
        visual_processed = layers.Dropout(self.model_config["dropout_rate"] * 0.5)(visual_processed)

        # === NUMERICAL BRANCH ===
        # Process numerical features (3 features: distance_to_prev_high, distance_to_prev_low, prev_day_range)
        numerical_features = layers.Dense(64, activation="relu", name="numerical_dense_1")(numerical_input)
        numerical_features = layers.Dropout(self.model_config["dropout_rate"])(numerical_features)
        
        numerical_features = layers.Dense(32, activation="relu", name="numerical_dense_2")(numerical_features)
        numerical_features = layers.Dropout(self.model_config["dropout_rate"])(numerical_features)
        
        numerical_features = layers.Dense(16, activation="relu", name="numerical_dense_3")(numerical_features)

        # === FEATURE FUSION ===
        # Concatenate visual and numerical features
        fused_features = layers.Concatenate(name="feature_fusion")([visual_processed, numerical_features])

        # Fusion processing layers
        x = layers.Dense(
            classification_head_size, activation="gelu", name="fusion_dense_1"
        )(fused_features)
        x = layers.Dropout(self.model_config["dropout_rate"])(x)

        x = layers.Dense(
            classification_head_size // 2, activation="gelu", name="fusion_dense_2"
        )(x)
        x = layers.Dropout(self.model_config["dropout_rate"] * 0.5)(x)

        # Final 4-class classification layer
        predictions = layers.Dense(
            num_classes, activation="softmax", name="daily_pattern_classification"
        )(x)

        # Create hybrid model
        self.model = keras.Model(
            inputs=[image_input, numerical_input], 
            outputs=predictions, 
            name="hybrid_daily_vit"
        )

        total_params = self.model.count_params()
        self.logger.info(
            f"Hybrid ViT model created with {total_params:,} parameters"
        )
        self.logger.info(f"Model supports 4-class classification with visual + numerical fusion")

        return self.model

    def compile_model(self, learning_rate: float = None, class_weights: Dict[int, float] = None):
        """
        Compile the hybrid ViT model.

        Args:
            learning_rate: Learning rate for optimizer
            class_weights: Class weights for handling imbalance
        """
        if self.model is None:
            raise ValueError("Model must be created before compilation")

        if learning_rate is None:
            learning_rate = self.model_config["learning_rate"]

        # Learning rate schedule for 4-class classification
        if self.config["training"].get("lr_scheduling", True):
            lr_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=1000,  # Will be updated based on dataset size
                alpha=0.1,
            )
        else:
            lr_schedule = learning_rate

        # Optimizer optimized for hybrid classification
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=self.model_config["weight_decay"]
        )

        # 4-class classification metrics (simplified for sparse labels)
        metrics = [
            "accuracy",
            "sparse_categorical_accuracy",
        ]

        # Loss function for 4-class classification (labels are 0-3 in TensorFlow)
        loss = "sparse_categorical_crossentropy"

        # Compile model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.logger.info("Hybrid ViT model compiled successfully")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Weight decay: {self.model_config['weight_decay']}")
        self.logger.info(f"Loss function: {loss}")
        self.logger.info(f"Classes: 4 (High Breakout, Low Breakdown, Range Expansion, Range Bound)")

        if class_weights:
            self.logger.info(f"Class weights: {class_weights}")

    def create_class_weights(
        self, class_distribution: Dict[int, int]
    ) -> Dict[int, float]:
        """
        Create class weights for handling class imbalance in 4-class system.

        Args:
            class_distribution: Dictionary with class counts (1-4)

        Returns:
            Dictionary with class weights (0-3 for TensorFlow)
        """
        total_samples = sum(class_distribution.values())
        num_classes = len(class_distribution)

        class_weights = {}
        for class_id, count in class_distribution.items():
            # Convert from 1-4 to 0-3 for TensorFlow
            tf_class_id = class_id - 1
            # Inverse frequency weighting
            class_weights[tf_class_id] = total_samples / (num_classes * count)

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
            filepath = f"{self.config['paths']['metadata']}/hybrid_vit_architecture_{timestamp}.png"

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

        self.logger.info(f"Hybrid model architecture saved: {filepath}")
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
                f"{self.config['paths']['models']}/hybrid_vit_weights_{timestamp}.h5"
            )

        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self.model.save_weights(weights_path)
        self.logger.info(f"Hybrid model weights saved: {weights_path}")
        return weights_path

    def predict_pattern(self, image_batch: np.ndarray, numerical_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict daily patterns from chart images and numerical features.

        Args:
            image_batch: Batch of chart images
            numerical_batch: Batch of numerical features

        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.model is None:
            raise ValueError("Model must be created and trained before prediction")

        # Get raw predictions
        predictions = self.model.predict([image_batch, numerical_batch], verbose=0)

        # Extract class predictions and confidence
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)

        # Convert to pattern labels (0-3 -> 1-4)
        pattern_labels = predicted_classes + 1

        # Pattern names
        pattern_names = {
            1: "High Breakout",
            2: "Low Breakdown", 
            3: "Range Expansion",
            4: "Range Bound"
        }
        
        pattern_labels_text = [pattern_names[label] for label in pattern_labels]

        return {
            "predictions": predictions,
            "predicted_classes": predicted_classes,  # 0-3 for TensorFlow
            "pattern_labels": pattern_labels,  # 1-4 for interpretation
            "pattern_names": pattern_labels_text,
            "confidence_scores": confidence_scores,
            "class_probabilities": {
                "high_breakout": predictions[:, 0],
                "low_breakdown": predictions[:, 1],
                "range_expansion": predictions[:, 2],
                "range_bound": predictions[:, 3],
            },
        }


def main():
    """Test the hybrid ViT model."""
    print("🚀 NQ_AI Hybrid ViT Model")
    print("4-Class Previous Day Levels Classification System")

    try:
        # Initialize model
        model_builder = HybridViTModel()

        # Create model
        print("🏗️  Creating hybrid ViT model...")
        model = model_builder.create_model()

        # Compile model
        print("⚙️  Compiling hybrid model...")
        model_builder.compile_model()

        # Print model summary
        print("📊 Hybrid Model Summary:")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Visual input shape: {model.input[0].shape}")
        print(f"   Numerical input shape: {model.input[1].shape}")
        print(f"   Output shape: {model.output.shape}")
        print(f"   Classes: 4")

        # Test with dummy data
        print("🧪 Testing with dummy data...")

        batch_size = 4
        image_size = model_builder.model_config["image_size"]
        dummy_images = np.random.rand(batch_size, image_size, image_size, 3)
        dummy_numerical = np.random.rand(batch_size, 3)  # 3 numerical features

        # Test prediction
        predictions = model.predict([dummy_images, dummy_numerical], verbose=0)
        pattern_results = model_builder.predict_pattern(dummy_images, dummy_numerical)

        print(f"✅ Hybrid ViT test successful!")
        print(f"   Visual input shape: {dummy_images.shape}")
        print(f"   Numerical input shape: {dummy_numerical.shape}")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[0]}")
        print(f"   Pattern names: {pattern_results['pattern_names']}")
        print(f"   Confidence scores: {pattern_results['confidence_scores']}")
        print(f"   Model size: ~{model.count_params() * 4 / 1e6:.1f} MB")

        print("\n🎯 Hybrid Model Features:")
        print("   - 4-class previous day levels classification")
        print("   - Hybrid architecture: ViT-Base + numerical features")
        print("   - Visual input: 224x224x3 chart images with reference lines")
        print("   - Numerical input: 3 key features (distance_to_prev_high, distance_to_prev_low, prev_day_range)")
        print("   - Feature fusion: Early fusion after separate processing")
        print("   - 87M+ parameter ViT-Base backbone")
        print("   - Multi-instrument training ready")
        print("   - Previous day level interaction analysis")

        # Test class weights calculation
        print("\n📊 Testing class weights calculation...")
        test_distribution = {1: 2400, 2: 2100, 3: 1800, 4: 2700}  # 4-class distribution
        class_weights = model_builder.create_class_weights(test_distribution)
        print(f"   Class weights: {class_weights}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
