#!/usr/bin/env python3
"""
Training Script for Pure Visual ViT-Base Model
No numerical features - images only
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from proper_vit_visual_model import create_proper_vit_model, compile_model
from visual_data_loader import VisualDataLoader


class VisualViTTrainer:
    """Trainer for pure visual ViT-Base model."""

    def __init__(self, config_path: str = "config/config_vit_base.yaml"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_gpu()

        # Initialize components
        self.data_loader = None
        self.model = None
        self.history = None

        logging.info("Visual ViT Trainer initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(
            logging, self.config.get("logging", {}).get("level", "INFO")
        )
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("visual_vit_training.log"),
            ],
        )

    def setup_gpu(self):
        """Setup GPU configuration."""
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                logging.error(f"GPU setup error: {e}")
        else:
            logging.info("No GPUs found, using CPU")

    def prepare_data(self):
        """Prepare data loaders."""
        logging.info("Preparing visual data...")

        self.data_loader = VisualDataLoader(
            data_root=self.config["paths"]["data_root"],
            image_size=224,  # ViT-Base standard
            batch_size=self.config["model"]["batch_size"],
            validation_split=0.2,
            test_split=0.2,
            random_state=42,
        )

        # Get datasets
        self.train_dataset = self.data_loader.get_train_dataset()
        self.val_dataset = self.data_loader.get_val_dataset()
        self.test_dataset = self.data_loader.get_test_dataset()

        logging.info("Data preparation complete")

    def create_model(self):
        """Create and compile the visual model."""
        logging.info("Creating pure visual ViT-Base model...")

        # Create model
        self.model = create_proper_vit_model(image_size=224, num_classes=4, dropout=0.1)

        # Compile model
        self.model = compile_model(
            self.model,
            learning_rate=3e-4,
            class_weights=self.data_loader.get_class_weights(),
        )

        logging.info("Model creation complete")

    def setup_callbacks(self, epochs: int):
        """Setup training callbacks."""
        callbacks = []

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
        )
        callbacks.append(early_stopping)

        # Learning rate reduction
        lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        )
        callbacks.append(lr_reduction)

        # Model checkpointing
        checkpoint_dir = Path(self.config["paths"]["checkpoints"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "visual_vit_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        callbacks.append(checkpoint)

        # TensorBoard logging
        tensorboard_dir = Path(self.config["paths"]["tensorboard_logs"])
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(
                tensorboard_dir
                / f"visual_vit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        )
        callbacks.append(tensorboard)

        return callbacks

    def train(self, epochs: int = 30):
        """Train the model."""
        logging.info(f"Starting training for {epochs} epochs...")

        callbacks = self.setup_callbacks(epochs)

        # Calculate steps per epoch
        train_steps = len(self.data_loader.train_samples) // self.data_loader.batch_size
        val_steps = len(self.data_loader.val_samples) // self.data_loader.batch_size

        logging.info(f"Training steps per epoch: {train_steps}")
        logging.info(f"Validation steps per epoch: {val_steps}")

        # Train model
        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1,
        )

        logging.info("Training completed!")

        return self.history

    def evaluate_model(self):
        """Evaluate model on test set."""
        logging.info("Evaluating model on test set...")

        # Basic evaluation
        test_results = self.model.evaluate(self.test_dataset, verbose=1)

        # Detailed predictions for analysis
        logging.info("Generating predictions for detailed analysis...")
        predictions = []
        true_labels = []

        for batch_images, batch_labels in self.test_dataset:
            batch_predictions = self.model.predict(batch_images, verbose=0)
            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels.numpy())

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        predicted_labels = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = np.mean(predicted_labels == true_labels)
        f1_macro = f1_score(true_labels, predicted_labels, average="macro")
        f1_weighted = f1_score(true_labels, predicted_labels, average="weighted")

        # Classification report
        class_names = [
            "High Breakout",
            "Low Breakdown",
            "Range Expansion",
            "Range Bound",
        ]
        class_report = classification_report(
            true_labels, predicted_labels, target_names=class_names, output_dict=True
        )

        # Results summary
        results = {
            "test_accuracy": float(accuracy),
            "test_loss": float(test_results[0]),
            "f1_score_macro": float(f1_macro),
            "f1_score_weighted": float(f1_weighted),
            "top_2_accuracy": float(test_results[2]) if len(test_results) > 2 else 0.0,
            "classification_report": class_report,
            "confusion_matrix": confusion_matrix(
                true_labels, predicted_labels
            ).tolist(),
            "num_test_samples": len(true_labels),
        }

        # Log results
        logging.info("EVALUATION RESULTS:")
        logging.info(f"  📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logging.info(f"  📈 Test Loss: {test_results[0]:.4f}")
        logging.info(f"  🎯 F1 Score (Macro): {f1_macro:.4f}")
        logging.info(f"  ⚖️  F1 Score (Weighted): {f1_weighted:.4f}")
        if len(test_results) > 2:
            logging.info(
                f"  📊 Top-2 Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.2f}%)"
            )

        # Per-class performance
        logging.info("Per-class performance:")
        for i, class_name in enumerate(class_names):
            class_metrics = class_report[str(i)]
            logging.info(
                f"  {class_name}: P={class_metrics['precision']:.3f}, R={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}"
            )

        return results

    def save_results(self, results: dict):
        """Save training and evaluation results."""
        output_dir = Path(self.config["paths"]["outputs"])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results
        results_file = output_dir / f"visual_vit_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save model
        model_file = output_dir / f"visual_vit_model_{timestamp}.keras"
        self.model.save(model_file)

        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Model saved to: {model_file}")

    def run_complete_training(self, epochs: int = 30):
        """Run complete training pipeline."""
        logging.info(
            "======================================================================"
        )
        logging.info("STARTING PURE VISUAL VIT-BASE TRAINING")
        logging.info(
            "======================================================================"
        )

        try:
            # Prepare data
            self.prepare_data()

            # Create model
            self.create_model()

            # Train model
            history = self.train(epochs)

            # Evaluate model
            results = self.evaluate_model()

            # Add training history to results
            results["training_history"] = {
                "loss": [float(x) for x in history.history["loss"]],
                "accuracy": [float(x) for x in history.history["accuracy"]],
                "val_loss": [float(x) for x in history.history["val_loss"]],
                "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
            }

            # Save results
            self.save_results(results)

            logging.info("Training pipeline completed successfully!")
            return results

        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Train Pure Visual ViT-Base Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_vit_base.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )

    args = parser.parse_args()

    # Override config with command line arguments if provided
    trainer = VisualViTTrainer(args.config)
    if hasattr(args, "batch_size"):
        trainer.config["model"]["batch_size"] = args.batch_size

    # Run training
    results = trainer.run_complete_training(args.epochs)

    # Print final summary
    print("\n" + "=" * 70)
    print("PURE VISUAL VIT-BASE TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Final Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)"
    )
    print(f"F1 Score (Weighted): {results['f1_score_weighted']:.4f}")
    if results.get("top_2_accuracy"):
        print(
            f"Top-2 Accuracy: {results['top_2_accuracy']:.4f} ({results['top_2_accuracy']*100:.2f}%)"
        )

    # Check if target accuracy achieved
    if results["test_accuracy"] >= 0.70:
        print("🎉 TARGET ACCURACY ACHIEVED! (≥70%)")
    else:
        print(
            f"❌ Target accuracy not reached. Got {results['test_accuracy']*100:.2f}%, need ≥70%"
        )


if __name__ == "__main__":
    main()
