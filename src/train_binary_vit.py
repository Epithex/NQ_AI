#!/usr/bin/env python3
"""
Training Script for Binary Visual ViT-Base Model
Trains 87M parameter ViT-Base for bullish/bearish pattern classification
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

from binary_vit_model import BinaryViTModel
from binary_data_loader import BinaryDataLoader

class BinaryViTTrainer:
    """Trainer for binary visual ViT-Base model."""
    
    def __init__(self, config_path: str = "config/config_binary_visual.yaml"):
        self.config = self._load_config(config_path)
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
            'best_epoch': 0,
            'best_val_f1': 0.0,
            'final_metrics': {}
        }
        
        logging.info("Binary ViT Trainer initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
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
                logging.FileHandler(f"{log_dir}/binary_training.log"),
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
        """Prepare datasets for training."""
        try:
            self.logger.info("Preparing binary classification datasets")
            
            # Initialize data loader
            self.data_loader = BinaryDataLoader(self.config_path)
            
            # Prepare datasets
            self.train_dataset, self.val_dataset, self.test_dataset = self.data_loader.prepare_datasets()
            
            # Get class weights
            self.class_weights = self.data_loader.class_weights
            
            # Validate datasets
            if not self.data_loader.validate_dataset(self.train_dataset):
                raise ValueError("Training dataset validation failed")
            
            if not self.data_loader.validate_dataset(self.val_dataset):
                raise ValueError("Validation dataset validation failed")
            
            self.logger.info("Dataset preparation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
    
    def create_model(self):
        """Create and compile the binary ViT model."""
        try:
            self.logger.info("Creating binary ViT-Base model")
            
            # Initialize model builder
            self.model_builder = BinaryViTModel(self.config_path)
            
            # Create model architecture
            self.model = self.model_builder.create_model()
            
            # Compile model
            self.model_builder.compile_model()
            
            # Log model information
            total_params = self.model.count_params()
            self.logger.info(f"Binary ViT model created: {total_params:,} parameters")
            
            # Save model architecture
            arch_path = self.model_builder.save_model_architecture()
            self.logger.info(f"Model architecture saved: {arch_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            raise
    
    def create_callbacks(self) -> list:
        """Create training callbacks."""
        callbacks = []
        
        try:
            # Checkpoint callback
            if self.config['training'].get('checkpoint_frequency', 0) > 0:
                checkpoint_dir = self.config['paths']['checkpoints']
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, 'binary_vit_epoch_{epoch:02d}_f1_{val_f1_score:.4f}.h5'),
                    monitor='val_f1_score',
                    mode='max',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )
                callbacks.append(checkpoint_callback)
            
            # Early stopping
            if self.config['training'].get('early_stopping', True):
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_f1_score',
                    mode='max',
                    patience=self.config['training'].get('early_stopping_patience', 8),
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(early_stopping)
            
            # Learning rate scheduling
            if self.config['training'].get('lr_scheduling', True):
                lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_f1_score',
                    mode='max',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
                callbacks.append(lr_schedule)
            
            # TensorBoard
            tensorboard_dir = self.config['paths']['tensorboard_logs']
            os.makedirs(tensorboard_dir, exist_ok=True)
            
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(tensorboard_dir, f"binary_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch'
            )
            callbacks.append(tensorboard_callback)
            
            # Custom metrics callback
            metrics_callback = BinaryMetricsCallback(self.logger)
            callbacks.append(metrics_callback)
            
            self.logger.info(f"Created {len(callbacks)} training callbacks")
            return callbacks
            
        except Exception as e:
            self.logger.error(f"Error creating callbacks: {e}")
            raise
    
    def train_model(self):
        """Train the binary ViT model."""
        try:
            self.logger.info("Starting binary ViT training")
            self.training_results['start_time'] = datetime.now().isoformat()
            
            # Get training parameters
            epochs = self.config['model']['epochs']
            batch_size = self.config['model']['batch_size']
            
            # Calculate steps per epoch
            train_steps = self.data_loader.get_steps_per_epoch('train')
            val_steps = self.data_loader.get_steps_per_epoch('val')
            
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  Epochs: {epochs}")
            self.logger.info(f"  Batch size: {batch_size}")
            self.logger.info(f"  Train steps per epoch: {train_steps}")
            self.logger.info(f"  Validation steps per epoch: {val_steps}")
            
            # Create callbacks
            callbacks = self.create_callbacks()
            
            # Train model
            self.history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                steps_per_epoch=train_steps,
                validation_data=self.val_dataset,
                validation_steps=val_steps,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            self.training_results['end_time'] = datetime.now().isoformat()
            self.training_results['total_epochs'] = len(self.history.history['loss'])
            
            # Find best epoch
            val_f1_scores = self.history.history.get('val_f1_score', [])\n            if val_f1_scores:
                best_epoch = np.argmax(val_f1_scores)
                self.training_results['best_epoch'] = best_epoch + 1
                self.training_results['best_val_f1'] = float(val_f1_scores[best_epoch])
            
            self.logger.info("Training completed successfully")
            self.logger.info(f"Best epoch: {self.training_results['best_epoch']}")
            self.logger.info(f"Best validation F1: {self.training_results['best_val_f1']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def evaluate_model(self):
        """Evaluate the trained model on test set."""
        try:
            self.logger.info("Evaluating binary ViT model on test set")
            
            # Evaluate on test set
            test_steps = self.data_loader.get_steps_per_epoch('test')
            test_results = self.model.evaluate(
                self.test_dataset,
                steps=test_steps,
                verbose=1
            )
            
            # Get metric names and values
            metric_names = self.model.metrics_names
            test_metrics = dict(zip(metric_names, test_results))
            
            self.training_results['final_metrics'] = test_metrics
            
            self.logger.info("Test set evaluation:")
            for metric, value in test_metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")
            
            # Generate detailed predictions for analysis
            self.generate_detailed_evaluation()
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def generate_detailed_evaluation(self):
        """Generate detailed evaluation metrics and visualizations."""
        try:
            self.logger.info("Generating detailed evaluation")
            
            # Get predictions on test set
            test_steps = self.data_loader.get_steps_per_epoch('test')
            predictions = self.model.predict(self.test_dataset, steps=test_steps, verbose=1)
            
            # Get true labels
            true_labels = []
            for _, labels in self.test_dataset.take(test_steps):
                true_labels.extend(labels.numpy())
            
            true_labels = np.array(true_labels)
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Generate classification report
            label_names = [self.config['classification']['labels'][i] for i in range(3)]
            class_report = classification_report(
                true_labels, predicted_labels, 
                target_names=label_names,
                output_dict=True
            )
            
            # Generate confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            
            # Save results
            self.save_evaluation_results(class_report, cm, label_names)
            
            # Create visualizations
            self.create_evaluation_plots(cm, label_names)
            
        except Exception as e:
            self.logger.error(f"Error generating detailed evaluation: {e}")
            raise
    
    def save_evaluation_results(self, class_report: dict, confusion_matrix: np.ndarray, 
                               label_names: list):
        """Save evaluation results to files."""
        try:
            output_dir = self.config['paths']['outputs']
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save training results
            results_file = os.path.join(output_dir, f'binary_training_results_{timestamp}.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'training_results': self.training_results,
                    'classification_report': class_report,
                    'confusion_matrix': confusion_matrix.tolist(),
                    'label_names': label_names
                }, f, indent=2, default=str)
            
            self.logger.info(f"Evaluation results saved: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")
            raise
    
    def create_evaluation_plots(self, confusion_matrix: np.ndarray, label_names: list):
        """Create evaluation visualizations."""
        try:
            output_dir = self.config['paths']['outputs']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Confusion matrix heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names
            )
            plt.title('Binary Classification Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_file = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Training history plots
            if self.history:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # Loss
                axes[0,0].plot(self.history.history['loss'], label='Training Loss')
                axes[0,0].plot(self.history.history['val_loss'], label='Validation Loss')
                axes[0,0].set_title('Model Loss')
                axes[0,0].set_xlabel('Epoch')
                axes[0,0].set_ylabel('Loss')
                axes[0,0].legend()
                
                # Accuracy
                axes[0,1].plot(self.history.history['accuracy'], label='Training Accuracy')
                axes[0,1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
                axes[0,1].set_title('Model Accuracy')
                axes[0,1].set_xlabel('Epoch')
                axes[0,1].set_ylabel('Accuracy')
                axes[0,1].legend()
                
                # F1 Score
                if 'f1_score' in self.history.history:
                    axes[1,0].plot(self.history.history['f1_score'], label='Training F1')
                    axes[1,0].plot(self.history.history['val_f1_score'], label='Validation F1')
                    axes[1,0].set_title('Model F1 Score')
                    axes[1,0].set_xlabel('Epoch')
                    axes[1,0].set_ylabel('F1 Score')
                    axes[1,0].legend()
                
                # Learning Rate
                if 'lr' in self.history.history:
                    axes[1,1].plot(self.history.history['lr'])
                    axes[1,1].set_title('Learning Rate')
                    axes[1,1].set_xlabel('Epoch')
                    axes[1,1].set_ylabel('Learning Rate')
                
                plt.tight_layout()
                
                history_file = os.path.join(output_dir, f'training_history_{timestamp}.png')
                plt.savefig(history_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Evaluation plots saved in: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating evaluation plots: {e}")
            raise
    
    def save_model(self):
        """Save the trained model."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save weights
            weights_path = self.model_builder.save_weights(
                f"{self.config['paths']['models']}/binary_vit_final_weights_{timestamp}.h5"
            )
            
            # Save complete model
            model_path = os.path.join(
                self.config['paths']['models'], 
                f'binary_vit_complete_model_{timestamp}'
            )
            self.model.save(model_path)
            
            self.logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

class BinaryMetricsCallback(tf.keras.callbacks.Callback):
    """Custom callback for logging binary classification metrics."""
    
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if logs:
            self.logger.info(f"Epoch {epoch + 1} - " + 
                           ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Binary ViT Model')
    parser.add_argument('--config', type=str, default='config/config_binary_visual.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to model weights to resume training')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting Binary ViT Training Pipeline")
        print(f"   Configuration: {args.config}")
        
        # Initialize trainer
        trainer = BinaryViTTrainer(args.config)
        
        # Prepare data
        print("📊 Preparing datasets...")
        trainer.prepare_data()
        
        # Create model
        print("🏗️  Creating binary ViT model...")
        trainer.create_model()
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"🔄 Resuming from: {args.resume}")
            trainer.model.load_weights(args.resume)
        
        # Train model
        print("🎯 Starting training...")
        trainer.train_model()
        
        # Evaluate model
        print("📈 Evaluating model...")
        trainer.evaluate_model()
        
        # Save model
        print("💾 Saving model...")
        trainer.save_model()
        
        print("✅ Binary ViT training completed successfully!")
        print(f"   Best F1 Score: {trainer.training_results['best_val_f1']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())