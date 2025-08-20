#!/usr/bin/env python3
"""
Training Script for Pure Visual Daily ViT Model
Trains pure visual ViT model for previous day levels pattern classification
Uses ONLY 448x448 chart images - NO numerical features - NO early stopping
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
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from pure_visual_daily_vit_model import PureVisualDailyViTModel
from pure_visual_data_loader import PureVisualDataLoader

class PureVisualDailyTrainer:
    """Trainer for pure visual daily ViT model with single input (images only)."""
    
    def __init__(self, config_path: str = "config/config_pure_visual_daily.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.validate_config()
        self.setup_logging()
        self.setup_gpu()
        
        # Initialize components
        self.data_loader = None
        self.model_builder = None
        self.model = None
        self.history = None
        
        # Training state
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
        
        # Results tracking
        self.training_results = {
            'start_time': None,
            'end_time': None,
            'total_epochs': 0,
            'final_metrics': {}
        }
        
        logging.info("Pure Visual Daily ViT Trainer initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def validate_config(self):
        """Validate configuration to prevent training failures."""
        try:
            # Check required sections
            required_sections = ['model', 'training', 'paths']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required config section: {section}")
            
            # Validate model parameters
            model_config = self.config['model']
            required_model_params = ['batch_size', 'learning_rate', 'epochs', 'weight_decay']
            for param in required_model_params:
                if param not in model_config:
                    raise ValueError(f"Missing required model parameter: {param}")
            
            # Validate training parameters
            training_config = self.config['training']
            
            # Check scheduler configuration
            if training_config.get('lr_scheduling', True):
                if training_config.get('lr_schedule') == 'cosine_annealing':
                    t_max = training_config.get('lr_scheduler_t_max', model_config['epochs'])
                    min_lr = training_config.get('lr_scheduler_min_lr', 1e-5)
                    if t_max <= 0:
                        raise ValueError(f"Invalid T_max for scheduler: {t_max}")
                    if min_lr <= 0 or min_lr >= model_config['learning_rate']:
                        raise ValueError(f"Invalid min_lr for scheduler: {min_lr}")
            
            # Check early stopping configuration
            if training_config.get('early_stopping', False):
                patience = training_config.get('early_stopping_patience', 50)
                if patience <= 0:
                    raise ValueError(f"Invalid early stopping patience: {patience}")
            
            # Validate batch size
            if model_config['batch_size'] <= 0:
                raise ValueError(f"Invalid batch size: {model_config['batch_size']}")
            
            # Validate epochs
            if model_config['epochs'] <= 0:
                raise ValueError(f"Invalid epochs: {model_config['epochs']}")
            
            # Validate learning rate
            if model_config['learning_rate'] <= 0:
                raise ValueError(f"Invalid learning rate: {model_config['learning_rate']}")
                
            print("✅ Configuration validation passed")
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
        
        # Create logs directory
        log_dir = self.config['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/pure_visual_daily_training.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def setup_gpu(self):
        """Setup GPU configuration for training."""
        try:
            # Configure GPU memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            else:
                self.logger.warning("No GPUs found, using CPU")
            
            # Mixed precision training
            if self.config['training'].get('mixed_precision', False):
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                self.logger.info("Mixed precision training enabled")
                
        except Exception as e:
            self.logger.error(f"Error setting up GPU: {e}")
    
    def prepare_data(self):
        """Prepare pure visual datasets for training."""
        try:
            self.logger.info("Preparing pure visual datasets...")
            
            # Initialize data loader
            self.data_loader = PureVisualDataLoader(self.config_path)
            
            # Prepare datasets
            self.train_dataset, self.val_dataset, self.test_dataset = (
                self.data_loader.prepare_datasets()
            )
            
            # Get class weights
            self.class_weights = self.data_loader.class_weights
            
            # Convert class weights from 1-4 to 0-3 for TensorFlow
            if self.class_weights:
                tf_class_weights = {}
                for class_id, weight in self.class_weights.items():
                    tf_class_weights[class_id - 1] = weight
                self.class_weights = tf_class_weights
            
            self.logger.info("Pure visual data preparation completed")
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
    
    def build_model(self):
        """Build and compile pure visual ViT model."""
        try:
            self.logger.info("Building pure visual ViT model...")
            
            # Initialize model builder
            self.model_builder = PureVisualDailyViTModel(self.config_path)
            
            # Create model
            self.model = self.model_builder.create_model()
            
            # Compile model
            self.model_builder.compile_model(class_weights=self.class_weights)
            
            self.logger.info(f"Pure visual model built with {self.model.count_params():,} parameters")
            self.logger.info("Model compiled successfully")
            
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            raise
    
    def create_callbacks(self):
        """Create training callbacks (NO early stopping as requested)."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_dir = self.config['paths']['checkpoints']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Determine save frequency based on configuration
        save_freq = 'epoch'  # Default to every epoch
        if self.config['training'].get('save_best_only', False):
            # Save only best model based on validation accuracy
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'pure_visual_daily_best_model.weights.h5'),
                save_weights_only=True,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        else:
            # Save every N epochs
            save_freq = self.config['training']['checkpoint_frequency']
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'pure_visual_daily_model_{epoch:02d}.weights.h5'),
                save_weights_only=True,
                save_freq=save_freq,
                verbose=1
            )
        callbacks.append(checkpoint_callback)
        
        # TensorBoard callback
        tensorboard_dir = self.config['paths']['tensorboard_logs']
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Simplified gradient norm tracking (if enabled)
        if self.config['training'].get('track_gradient_norms', True):
            # Use a simple custom callback that tracks training metrics
            class SimpleMetricsCallback(tf.keras.callbacks.Callback):
                def __init__(self, logger):
                    super().__init__()
                    self.logger = logger
                    
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        lr = self.model.optimizer.learning_rate
                        if hasattr(lr, 'numpy'):
                            current_lr = float(lr.numpy())
                        else:
                            current_lr = float(lr)
                        self.logger.info(f"Epoch {epoch + 1}: lr={current_lr:.6f}, loss={logs.get('loss', 0):.4f}, val_acc={logs.get('val_accuracy', 0):.4f}")
                        
                        # Log additional metrics to TensorBoard via logs
                        logs['learning_rate'] = current_lr
            
            metrics_callback = SimpleMetricsCallback(self.logger)
            callbacks.append(metrics_callback)
            self.logger.info("Added training metrics tracking callback")
        
        # Learning rate scheduler - ONLY CosineAnnealingLR
        if self.config['training'].get('lr_scheduling', True):
            scheduler_type = self.config['training'].get('lr_schedule', 'cosine_annealing')
            
            if scheduler_type == 'cosine_annealing':
                import math
                t_max = self.config['training'].get('lr_scheduler_t_max', self.config['model']['epochs'])
                min_lr = self.config['training'].get('lr_scheduler_min_lr', 1e-5)
                initial_lr = self.config['model']['learning_rate']
                
                def cosine_annealing_schedule(epoch, lr):
                    """CosineAnnealingLR implementation"""
                    return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / t_max)) / 2
                
                lr_callback = tf.keras.callbacks.LearningRateScheduler(
                    cosine_annealing_schedule,
                    verbose=1
                )
                callbacks.append(lr_callback)
                self.logger.info(f"Using CosineAnnealingLR scheduler: T_max={t_max}, min_lr={min_lr}, initial_lr={initial_lr}")
            else:
                self.logger.warning(f"Unsupported scheduler: {scheduler_type}. Using constant learning rate.")
        else:
            self.logger.info("Learning rate scheduling disabled - using constant learning rate")
        
        # Early stopping with 50 epoch patience
        if self.config['training'].get('early_stopping', False):
            patience = self.config['training'].get('early_stopping_patience', 50)
            monitor = self.config['training'].get('early_stopping_monitor', 'val_accuracy')
            mode = self.config['training'].get('early_stopping_mode', 'max')
            
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                mode=mode,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping_callback)
            self.logger.info(f"Added early stopping: monitor={monitor}, patience={patience}, mode={mode}")
        
        self.logger.info(f"Created {len(callbacks)} training callbacks")
        return callbacks
    
    def train_model(self):
        """Train the pure visual model."""
        try:
            self.logger.info("Starting pure visual model training...")
            self.training_results['start_time'] = datetime.now()
            
            # Get training parameters
            epochs = self.config['model']['epochs']
            steps_per_epoch = self.data_loader.get_steps_per_epoch('train')
            validation_steps = self.data_loader.get_steps_per_epoch('val')
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  Epochs: {epochs}")
            self.logger.info(f"  Steps per epoch: {steps_per_epoch}")
            self.logger.info(f"  Validation steps: {validation_steps}")
            self.logger.info(f"  Early stopping: DISABLED")
            
            # Create callbacks
            callbacks = self.create_callbacks()
            
            # Configure validation frequency
            validation_freq = self.config['training'].get('validation_frequency', 1)
            
            # Train model with validation monitoring
            self.history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.val_dataset,
                validation_steps=validation_steps,
                validation_freq=validation_freq,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            self.logger.info(f"Validation performed every {validation_freq} epoch(s)")
            
            self.training_results['end_time'] = datetime.now()
            self.training_results['total_epochs'] = epochs
            
            self.logger.info("Pure visual model training completed")
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def evaluate_model(self):
        """Evaluate the trained pure visual model."""
        try:
            self.logger.info("Evaluating pure visual model...")
            
            # Evaluate on test set
            test_steps = self.data_loader.get_steps_per_epoch('test')
            test_loss, test_accuracy = self.model.evaluate(
                self.test_dataset,
                steps=test_steps,
                verbose=1
            )
            
            self.logger.info(f"Test Results:")
            self.logger.info(f"  Test Loss: {test_loss:.4f}")
            self.logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
            
            # Generate predictions for detailed metrics
            self.logger.info("Generating predictions for detailed analysis...")
            
            predictions = []
            true_labels = []
            
            for batch_images, batch_labels in self.test_dataset.take(test_steps):
                batch_predictions = self.model.predict(batch_images, verbose=0)
                predictions.extend(np.argmax(batch_predictions, axis=1))
                true_labels.extend(batch_labels.numpy())
            
            # Calculate detailed metrics
            f1 = f1_score(true_labels, predictions, average='weighted')
            
            # Classification report
            pattern_names = ['High Breakout', 'Low Breakdown', 'Range Expansion', 'Range Bound']
            class_report = classification_report(
                true_labels, 
                predictions, 
                target_names=pattern_names,
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            
            # Store final metrics
            self.training_results['final_metrics'] = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': cm.tolist()
            }
            
            self.logger.info(f"Final Metrics:")
            self.logger.info(f"  F1 Score: {f1:.4f}")
            
            # Log class-wise performance
            for i, pattern_name in enumerate(pattern_names):
                precision = class_report[pattern_name]['precision']
                recall = class_report[pattern_name]['recall']
                f1_class = class_report[pattern_name]['f1-score']
                self.logger.info(f"  {pattern_name}: P={precision:.3f}, R={recall:.3f}, F1={f1_class:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def save_results(self):
        """Save training results and plots."""
        try:
            self.logger.info("Saving training results...")
            
            # Create outputs directory
            outputs_dir = self.config['paths']['outputs']
            os.makedirs(outputs_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save training results
            results_file = os.path.join(outputs_dir, f"pure_visual_daily_training_results_{timestamp}.json")
            with open(results_file, 'w') as f:
                json.dump(self.training_results, f, indent=2, default=str)
            
            self.logger.info(f"Training results saved: {results_file}")
            
            # Save training history plot
            if self.history:
                self.plot_training_history(outputs_dir, timestamp)
            
            # Save confusion matrix plot
            if 'confusion_matrix' in self.training_results['final_metrics']:
                self.plot_confusion_matrix(outputs_dir, timestamp)
            
            # Save final model weights
            model_file = os.path.join(outputs_dir, f"pure_visual_daily_model_{timestamp}.weights.h5")
            self.model.save_weights(model_file)
            self.logger.info(f"Model weights saved: {model_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def plot_training_history(self, outputs_dir: str, timestamp: str):
        """Plot training history."""
        try:
            plt.figure(figsize=(12, 4))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Pure Visual Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Pure Visual Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            plot_file = os.path.join(outputs_dir, f"training_history_pure_visual_daily_{timestamp}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training history plot saved: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")
    
    def plot_confusion_matrix(self, outputs_dir: str, timestamp: str):
        """Plot confusion matrix."""
        try:
            cm = np.array(self.training_results['final_metrics']['confusion_matrix'])
            pattern_names = ['High Breakout', 'Low Breakdown', 'Range Expansion', 'Range Bound']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=pattern_names,
                yticklabels=pattern_names
            )
            plt.title('Pure Visual Model Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plot_file = os.path.join(outputs_dir, f"confusion_matrix_pure_visual_daily_{timestamp}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix plot saved: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}")
    
    def run_training(self):
        """Run the complete pure visual training pipeline."""
        try:
            self.logger.info("Starting pure visual training pipeline...")
            
            # Prepare data
            self.prepare_data()
            
            # Build model
            self.build_model()
            
            # Train model
            self.train_model()
            
            # Evaluate model
            self.evaluate_model()
            
            # Save results
            self.save_results()
            
            self.logger.info("Pure visual training pipeline completed successfully!")
            
            # Print summary
            final_accuracy = self.training_results['final_metrics']['test_accuracy']
            final_f1 = self.training_results['final_metrics']['f1_score']
            total_epochs = self.training_results['total_epochs']
            
            print(f"\n🎯 Pure Visual Training Summary:")
            print(f"   Total Epochs: {total_epochs} (NO early stopping)")
            print(f"   Final Test Accuracy: {final_accuracy:.4f}")
            print(f"   Final F1 Score: {final_f1:.4f}")
            print(f"   Model Parameters: {self.model.count_params():,}")
            print(f"   Approach: Pure visual learning (NO numerical features)")
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Pure Visual Daily ViT Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_pure_visual_daily.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        print("🚀 NQ_AI Pure Visual Daily ViT Training")
        print("4-Class Previous Day Levels Classification - Pure Visual Learning")
        print("=" * 70)
        
        # Initialize trainer
        trainer = PureVisualDailyTrainer(args.config)
        
        # Run training
        trainer.run_training()
        
        print("\n✅ Pure visual training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())