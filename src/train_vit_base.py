#!/usr/bin/env python3
"""
ViT-Base Hybrid Model Training Script
Train 87M parameter ViT-Base model on daily NQ dataset
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import yaml

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vit_base_hybrid_model import ViTBaseHybridModel
from daily_data_loader import DailyDataLoader

class ViTBaseTrainer:
    """Trainer for ViT-Base hybrid model."""
    
    def __init__(self, config_path: str = "config/config_vit_base.yaml"):
        """Initialize trainer with configuration."""
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_paths()
        
        # Initialize components
        self.model_builder = ViTBaseHybridModel(config_path)
        self.data_loader = DailyDataLoader(config_path)
        
        # Training state
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.training_history = None
        
        # Generate timestamp for this training run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.logger.info("ViT-Base hybrid trainer initialized")
        self.logger.info(f"Configuration: {config_path}")
        self.logger.info(f"Training run ID: {self.timestamp}")
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging for training."""
        log_dir = self.config['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/train_vit_base.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_paths(self):
        """Setup and create necessary directories."""
        self.paths = self.config['paths']
        
        # Create directories
        for path_key, path_value in self.paths.items():
            if path_key.endswith('_dir') or path_key in ['checkpoints', 'outputs']:
                os.makedirs(path_value, exist_ok=True)
    
    def setup_gpu(self):
        """Configure GPU settings for training."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Enable memory growth to avoid OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                self.logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
                
                # Enable mixed precision for memory efficiency with large model
                if self.config['training'].get('mixed_precision', True):
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    self.logger.info("Mixed precision enabled (float16)")
                
            except RuntimeError as e:
                self.logger.warning(f"GPU configuration error: {e}")
        else:
            self.logger.info("No GPUs found, using CPU")
    
    def load_datasets(self):
        """Load and prepare datasets for training."""
        self.logger.info("Loading daily datasets...")
        
        # Load datasets using the daily data loader
        try:
            datasets = self.data_loader.load_datasets()
            
            self.train_dataset = datasets['train']
            self.val_dataset = datasets['validation']  
            self.test_dataset = datasets['test']
            
            # Get dataset information
            dataset_info = self.data_loader.get_dataset_info()
            
            self.logger.info("Dataset loading completed:")
            self.logger.info(f"  Training samples: {dataset_info['train_samples']}")
            self.logger.info(f"  Validation samples: {dataset_info['val_samples']}")
            self.logger.info(f"  Test samples: {dataset_info['test_samples']}")
            self.logger.info(f"  Label distribution: {dataset_info['label_distribution']}")
            
            return dataset_info
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise
    
    def create_model(self):
        """Create and compile the ViT-Base hybrid model."""
        self.logger.info("Creating ViT-Base hybrid model...")
        
        try:
            # Create the model
            self.model = self.model_builder.create_hybrid_model()
            
            # Compile the model
            self.model_builder.compile_model()
            
            # Log model information
            total_params = self.model.count_params()
            self.logger.info(f"Model created successfully:")
            self.logger.info(f"  Total parameters: {total_params:,}")
            self.logger.info(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB (float32)")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise
    
    def setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpoint - save best weights
        checkpoint_path = f"{self.paths['checkpoints']}/vit_base_best_{self.timestamp}.weights.h5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            mode='max'
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['training']['reduce_lr_patience'],
            min_lr=1e-8,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        tensorboard_dir = f"{self.paths['tensorboard_logs']}/vit_base_{self.timestamp}"
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,  # Disable image writing for large models
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        # CSV logging
        csv_logger = tf.keras.callbacks.CSVLogger(
            f"{self.paths['outputs']}/training_log_vit_base_{self.timestamp}.csv",
            append=True
        )
        callbacks.append(csv_logger)
        
        self.logger.info(f"Training callbacks configured:")
        self.logger.info(f"  Checkpoint: {checkpoint_path}")
        self.logger.info(f"  TensorBoard: {tensorboard_dir}")
        
        return callbacks
    
    def train_model(self, epochs: int = None) -> tf.keras.Model:
        """Train the ViT-Base hybrid model."""
        if epochs is None:
            epochs = self.config['model']['epochs']
        
        self.logger.info("=" * 60)
        self.logger.info("Starting ViT-Base Hybrid Model Training")
        self.logger.info("=" * 60)
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Batch size: {self.config['model']['batch_size']}")
        self.logger.info(f"Learning rate: {self.config['model']['learning_rate']}")
        
        try:
            # Setup callbacks
            callbacks = self.setup_callbacks()
            
            # Start training
            self.logger.info("Starting training with dynamic steps per epoch")
            
            self.training_history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                validation_data=self.val_dataset,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Training completed successfully!")
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate_model(self) -> Dict:
        """Evaluate the trained model on test set."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("Evaluating ViT-Base hybrid model on test set...")
        
        try:
            # Evaluate on test dataset
            test_results = self.model.evaluate(self.test_dataset, verbose=1)
            
            # Get predictions for detailed analysis
            self.logger.info("Generating predictions for analysis...")
            predictions = []
            true_labels = []
            
            for batch in self.test_dataset:
                (images, features), labels = batch
                batch_pred = self.model.predict([images, features], verbose=0)
                
                predictions.extend(np.argmax(batch_pred, axis=1))
                true_labels.extend(labels.numpy())
            
            # Convert to numpy arrays
            predictions = np.array(predictions)
            true_labels = np.array(true_labels)
            
            # Calculate detailed metrics
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            
            accuracy = accuracy_score(true_labels, predictions)
            class_report = classification_report(
                true_labels, 
                predictions,
                target_names=[f'Pattern {i}' for i in range(1, 5)],
                output_dict=True
            )
            conf_matrix = confusion_matrix(true_labels, predictions)
            
            # Compile results
            results = {
                'test_accuracy': accuracy,
                'test_loss': test_results[0],
                'test_metrics': {
                    metric_name: test_results[i+1] 
                    for i, metric_name in enumerate(['accuracy', 'top_2_accuracy', 'precision', 'recall', 'f1_score'])
                    if i+1 < len(test_results)
                },
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': predictions.tolist(),
                'true_labels': true_labels.tolist(),
                'model_parameters': self.model.count_params(),
                'training_timestamp': self.timestamp
            }
            
            self.logger.info("Evaluation completed:")
            self.logger.info(f"  Test Accuracy: {accuracy:.4f}")
            self.logger.info(f"  Test Loss: {test_results[0]:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def save_results(self, results: Dict):
        """Save training and evaluation results."""
        try:
            # Save training history
            if self.training_history:
                history_dict = {k: [float(v) for v in values] for k, values in self.training_history.history.items()}
                history_path = f"{self.paths['outputs']}/training_history_vit_base_{self.timestamp}.json"
                with open(history_path, 'w') as f:
                    json.dump(history_dict, f, indent=2)
                self.logger.info(f"Training history saved: {history_path}")
            
            # Save evaluation results
            results_path = f"{self.paths['outputs']}/evaluation_results_vit_base_{self.timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Evaluation results saved: {results_path}")
            
            # Save model summary
            summary_path = f"{self.paths['outputs']}/model_summary_vit_base_{self.timestamp}.txt"
            with open(summary_path, 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            self.logger.info(f"Model summary saved: {summary_path}")
            
            # Save configuration used for this run
            config_path = f"{self.paths['outputs']}/config_vit_base_{self.timestamp}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
    
    def run_complete_training(self, epochs: int = None):
        """Run complete training pipeline."""
        try:
            # Setup GPU
            self.setup_gpu()
            
            # Load datasets
            dataset_info = self.load_datasets()
            
            # Create model
            self.create_model()
            
            # Train model
            self.train_model(epochs)
            
            # Evaluate model
            results = self.evaluate_model()
            
            # Save results
            self.save_results(results)
            
            self.logger.info("=" * 60)
            self.logger.info("ViT-Base Hybrid Training Pipeline Completed Successfully!")
            self.logger.info("=" * 60)
            self.logger.info(f"Final test accuracy: {results['test_accuracy']:.4f}")
            self.logger.info(f"Model parameters: {results['model_parameters']:,}")
            self.logger.info(f"Training ID: {self.timestamp}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ViT-Base Hybrid Model')
    parser.add_argument('--config', type=str, default='config/config_vit_base.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--test-model', action='store_true',
                       help='Test model creation without training')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 NQ_AI ViT-Base Hybrid Model Training")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Test mode: {args.test_model}")
    
    try:
        # Initialize trainer
        trainer = ViTBaseTrainer(args.config)
        
        if args.test_model:
            # Test model creation only
            print("🧪 Testing model creation...")
            trainer.create_model()
            print("✅ Model creation test successful!")
            print(f"   Parameters: {trainer.model.count_params():,}")
            
            # Test with dummy data
            print("🧪 Testing model inference...")
            batch_size = 2
            image_size = trainer.config['model']['image_size']
            num_features = len(trainer.config['features']['feature_names'])
            
            dummy_images = np.random.rand(batch_size, image_size, image_size, 3)
            dummy_features = np.random.rand(batch_size, num_features)
            
            predictions = trainer.model.predict([dummy_images, dummy_features])
            print(f"✅ Model inference test successful!")
            print(f"   Input shapes: Images {dummy_images.shape}, Features {dummy_features.shape}")
            print(f"   Output shape: {predictions.shape}")
            
        else:
            # Run complete training
            results = trainer.run_complete_training(args.epochs)
            print("🎉 Training completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())