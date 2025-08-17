#!/usr/bin/env python3
"""
25M Parameter ViT Training Script for NQ_AI
Trains smaller ViT model on existing multi-instrument dataset
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleViT25M:
    """Simplified 25M parameter ViT for financial pattern classification."""

    def __init__(self, num_classes=4, image_size=224):
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = 16
        self.projection_dim = 512  # Smaller than full ViT-Base (768)
        self.num_heads = 8  # Fewer heads than ViT-Base (12)
        self.num_layers = 8  # Fewer layers than ViT-Base (12)
        self.mlp_dim = 2048  # Smaller MLP than ViT-Base (3072)

    def create_model(self):
        """Create 25M parameter ViT model."""
        inputs = tf.keras.layers.Input(shape=(self.image_size, self.image_size, 3))

        # Patch embedding
        num_patches = (self.image_size // self.patch_size) ** 2

        # Extract patches
        patches = tf.keras.layers.Conv2D(
            self.projection_dim, self.patch_size, strides=self.patch_size
        )(inputs)
        patches = tf.keras.layers.Reshape((num_patches, self.projection_dim))(patches)

        # Position embeddings
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=self.projection_dim
        )(positions)

        # Add class token
        class_token = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(tf.zeros_like(x[:, 0]), 1)
        )(patches)

        # Combine patches with position embeddings
        encoded_patches = patches + position_embedding
        encoded_patches = tf.keras.layers.Concatenate(axis=1)(
            [class_token, encoded_patches]
        )

        # Transformer blocks (8 layers for 25M params)
        for i in range(self.num_layers):
            # Layer normalization 1
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

            # Multi-head self-attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.projection_dim // self.num_heads,
                dropout=0.1,
            )(x1, x1)

            # Skip connection 1
            x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

            # Layer normalization 2
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

            # MLP
            x3 = tf.keras.layers.Dense(self.mlp_dim, activation="gelu")(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)
            x3 = tf.keras.layers.Dense(self.projection_dim)(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)

            # Skip connection 2
            encoded_patches = tf.keras.layers.Add()([x3, x2])

        # Classification head
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        x = x[:, 0]  # Use class token
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)

        # Print parameter count
        param_count = model.count_params()
        logger.info(f"Created 25M ViT model with {param_count:,} parameters")

        return model


def load_dataset(data_dir="data"):
    """Load the existing multi-instrument dataset from JSON labels and PNG images."""
    import json
    import glob
    from PIL import Image
    from sklearn.model_selection import train_test_split

    logger.info(f"Loading dataset from {data_dir}")

    # Load all samples
    label_files = glob.glob(f"{data_dir}/labels/daily_sample_*.json")

    images = []
    labels = []
    instruments = []
    dates = []

    logger.info(f"Found {len(label_files)} label files")

    for i, label_file in enumerate(label_files):
        try:
            # Load label
            with open(label_file, "r") as f:
                sample_data = json.load(f)

            # Extract info from filename: daily_sample_NQ_20060523.json
            filename = os.path.basename(label_file)
            parts = filename.replace(".json", "").split("_")
            instrument = parts[2]
            date = parts[3]

            # Find corresponding image
            image_file = f"{data_dir}/images/daily_chart_{instrument}_{date}.png"

            if os.path.exists(image_file):
                # Load and resize image
                img = Image.open(image_file)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0

                # Get pattern label
                pattern = sample_data.get(
                    "pattern_rank",
                    sample_data.get("pattern", sample_data.get("label", 0)),
                )

                if pattern in [1, 2, 3, 4]:  # Valid patterns
                    images.append(img_array)
                    labels.append(pattern - 1)  # Convert 1-4 to 0-3
                    instruments.append(instrument)
                    dates.append(date)

            if (i + 1) % 1000 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(label_files)} files, {len(images)} valid samples"
                )

        except Exception as e:
            logger.warning(f"Error processing {label_file}: {e}")
            continue

    logger.info(f"Loaded {len(images)} total samples")

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Class distribution:")
    for label, count in zip(unique, counts):
        percentage = count / len(labels) * 100
        logger.info(f"  Class {label + 1}: {count} samples ({percentage:.1f}%)")

    # Split data: 70% train, 15% val, 15% test
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    logger.info(f"Train: {len(train_images)} samples")
    logger.info(f"Val: {len(val_images)} samples")
    logger.info(f"Test: {len(test_images)} samples")

    # Convert labels to categorical
    train_labels = tf.keras.utils.to_categorical(train_labels, 4)
    val_labels = tf.keras.utils.to_categorical(val_labels, 4)
    test_labels = tf.keras.utils.to_categorical(test_labels, 4)

    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    )


def create_callbacks(output_dir):
    """Create training callbacks (NO early stopping)."""
    callbacks = []

    # Model checkpoint
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{output_dir}/vit_25m_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate scheduling
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
    )
    callbacks.append(lr_schedule)

    # CSV logging
    csv_logger = tf.keras.callbacks.CSVLogger(
        f"{output_dir}/training_log_25m.csv", append=True
    )
    callbacks.append(csv_logger)

    logger.info("Callbacks created (NO early stopping - full 30 epochs)")
    return callbacks


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("25M Parameter ViT Training (No Early Stopping)")
    logger.info("=" * 60)

    # Configuration
    config = {"batch_size": 8, "learning_rate": 5e-5, "epochs": 30, "num_classes": 4}

    logger.info(f"Configuration: {config}")

    # Create output directory
    output_dir = "models/outputs/vit_25m"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load dataset
        (
            (train_images, train_labels),
            (val_images, val_labels),
            (test_images, test_labels),
        ) = load_dataset()

        # Create model
        vit_model = SimpleViT25M(num_classes=4)
        model = vit_model.create_model()

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=config["learning_rate"], weight_decay=0.01
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Create callbacks
        callbacks = create_callbacks(output_dir)

        # Train model (NO early stopping)
        logger.info("Starting training for full 30 epochs...")
        start_time = datetime.now()

        history = model.fit(
            train_images,
            train_labels,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            verbose=1,
        )

        end_time = datetime.now()
        training_duration = end_time - start_time

        # Final evaluation
        logger.info("Evaluating on test set...")
        test_results = model.evaluate(test_images, test_labels, verbose=1)
        test_accuracy = test_results[1]
        test_loss = test_results[0]

        # Save final model
        model.save(f"{output_dir}/vit_25m_final.h5")

        # Save training history
        with open(f"{output_dir}/training_history.json", "w") as f:
            history_dict = {
                k: [float(x) for x in v] for k, v in history.history.items()
            }
            json.dump(history_dict, f, indent=2)

        # Save results summary
        results = {
            "model_type": "25M Parameter ViT",
            "parameters": model.count_params(),
            "training_duration": str(training_duration),
            "epochs_completed": config["epochs"],
            "final_test_accuracy": float(test_accuracy),
            "final_test_loss": float(test_loss),
            "early_stopping": False,
            "config": config,
        }

        with open(f"{output_dir}/results_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print final results
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Model: 25M Parameter ViT")
        logger.info(f"Parameters: {model.count_params():,}")
        logger.info(f"Training Duration: {training_duration}")
        logger.info(f"Epochs: {config['epochs']} (no early stopping)")
        logger.info(
            f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)"
        )
        logger.info(f"Final Test Loss: {test_loss:.4f}")
        logger.info(f"vs 87M Model Baseline: 42.46%")

        improvement = (test_accuracy - 0.4246) * 100
        if improvement > 0:
            logger.info(f"IMPROVEMENT: +{improvement:.2f}% over 87M model")
        else:
            logger.info(f"Performance: {improvement:.2f}% vs 87M model")

        logger.info(f"Model saved: {output_dir}/vit_25m_final.h5")

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()

    if results:
        print(f"\n🎯 25M ViT Training Complete!")
        print(f"   Accuracy: {results['final_test_accuracy']:.1%}")
        print(f"   Parameters: {results['parameters']:,}")
        print(f"   Duration: {results['training_duration']}")
        print(f"   Model: models/outputs/vit_25m/vit_25m_final.h5")
    else:
        print(f"\n❌ Training failed.")
        sys.exit(1)
