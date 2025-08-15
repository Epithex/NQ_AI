#!/usr/bin/env python3
"""
Daily NQ Pattern Analysis - Training Script
Complete training pipeline for hybrid ViT model
"""

import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml
import logging
import argparse

# Import our components
from hybrid_vit_model import HybridViTModel
from daily_data_loader import DailyDataLoader

class DailyModelTrainer:
    """Complete training pipeline for daily NQ pattern analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the trainer."""
        self.config = self.load_config(config_path)
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.paths = self.config['paths']
        
        self.setup_logging()
        self.setup_directories()
        
        # Components
        self.model_builder = None
        self.data_loader = None
        self.model = None
        
        # Training state
        self.training_history = None
        self.best_val_accuracy = 0.0
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = self.paths['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for this training run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/training_{self.timestamp}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create all required directories."""
        directories = [
            self.paths['models'],
            self.paths['outputs'],
            self.paths['checkpoints'],
            self.paths['tensorboard_logs'],
            self.paths['metadata']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_components(self):
        """Initialize model builder and data loader."""
        self.logger.info("Initializing training components...")
        
        # Initialize model builder
        self.model_builder = HybridViTModel()
        
        # Initialize data loader
        self.data_loader = DailyDataLoader()
        
        self.logger.info("Components initialized successfully")
    
    def prepare_data(self, dataset_name: str = "test_daily_nq"):
        """
        Prepare training data.
        
        Args:
            dataset_name: Name of dataset to load
        """
        self.logger.info(f"Preparing data: {dataset_name}")
        
        # Load dataset index
        samples = self.data_loader.load_dataset_index(dataset_name)
        self.logger.info(f"Loaded {len(samples)} total samples")
        
        # Split dataset (80% train, 10% validation, 10% test)
        train_samples, val_samples, test_samples = self.data_loader.split_dataset(
            test_size=0.1,
            val_size=0.111,  # 10% of remaining 90% = ~10% of total
            random_state=42
        )
        
        # Fit feature scaler on training data
        self.data_loader.fit_feature_scaler(train_samples)
        
        # Save scaler for later use
        scaler_path = self.data_loader.save_preprocessor_state(
            f"{self.paths['metadata']}/feature_scaler_{self.timestamp}.pkl"
        )
        
        # Create TensorFlow datasets
        self.train_dataset = self.data_loader.get_train_dataset()
        self.val_dataset = self.data_loader.get_val_dataset()
        self.test_dataset = self.data_loader.get_test_dataset()
        
        # Calculate class weights
        self.class_weights = self.data_loader.get_class_weights()
        
        self.logger.info("Data preparation completed")
        return {
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'scaler_path': scaler_path
        }
    
    def build_model(self):
        """Build and compile the hybrid ViT model."""
        self.logger.info("Building hybrid ViT model...")
        
        # Create model
        self.model = self.model_builder.create_hybrid_model()
        
        # Compile model
        self.model_builder.compile_model()
        
        # Log model summary
        self.logger.info("Model architecture:")
        self.logger.info(f"Total parameters: {self.model.count_params():,}")
        
        # Save model architecture
        arch_path = self.model_builder.save_model_architecture(
            f"{self.paths['metadata']}/model_architecture_{self.timestamp}.png"
        )
        
        self.logger.info("Model building completed")
        return arch_path
    
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = f"{self.paths['checkpoints']}/best_model_{self.timestamp}.weights.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # TensorBoard
        tensorboard_path = f"{self.paths['tensorboard_logs']}/training_{self.timestamp}"
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        # Custom callback for tracking metrics
        class MetricsLogger(tf.keras.callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                
                # Update best validation accuracy
                val_acc = logs.get('val_accuracy', 0)
                if val_acc > self.trainer.best_val_accuracy:
                    self.trainer.best_val_accuracy = val_acc
                    self.trainer.logger.info(f"New best validation accuracy: {val_acc:.4f}")
                
                # Log metrics
                self.trainer.logger.info(
                    f"Epoch {epoch + 1}: "
                    f"loss={logs.get('loss', 0):.4f}, "
                    f"acc={logs.get('accuracy', 0):.4f}, "
                    f"val_loss={logs.get('val_loss', 0):.4f}, "
                    f"val_acc={logs.get('val_accuracy', 0):.4f}"
                )
        
        callbacks.append(MetricsLogger(self))
        
        self.logger.info(f"Setup {len(callbacks)} training callbacks")
        return callbacks
    
    def train_model(self, epochs: int = None) -> Dict:
        """Train the model."""
        if epochs is None:
            epochs = self.model_config['epochs']
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Calculate approximate steps per epoch (batch size is handled by data loader)
        self.logger.info("Starting training with dynamic steps per epoch")
        
        # Start training
        start_time = datetime.now()
        
        try:
            self.training_history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                validation_data=self.val_dataset,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = datetime.now() - start_time
            self.logger.info(f"Training completed in {training_time}")
            
            return {
                'status': 'completed',
                'epochs_trained': len(self.training_history.history['loss']),
                'best_val_accuracy': self.best_val_accuracy,
                'training_time': str(training_time)
            }
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            return {
                'status': 'interrupted',
                'epochs_trained': len(self.training_history.history['loss']) if self.training_history else 0,
                'best_val_accuracy': self.best_val_accuracy
            }
        
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_model(self) -> Dict:
        """Evaluate model on test set."""
        self.logger.info("Evaluating model on test set...")
        
        try:
            # Evaluate on test set
            test_results = self.model.evaluate(
                self.test_dataset,
                verbose=1,
                return_dict=True
            )
            
            self.logger.info("Test evaluation completed")
            self.logger.info(f"Test accuracy: {test_results.get('accuracy', 0):.4f}")
            self.logger.info(f"Test loss: {test_results.get('loss', 0):.4f}")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {'error': str(e)}
    
    def generate_predictions(self, num_samples: int = 10) -> Dict:
        """Generate sample predictions."""
        self.logger.info(f"Generating predictions for {num_samples} samples...")
        
        try:
            # Get a batch of test data
            for batch in self.test_dataset.take(1):
                (images, features), true_labels = batch
                
                # Make predictions
                predictions = self.model.predict([images[:num_samples], features[:num_samples]])
                predicted_labels = np.argmax(predictions, axis=1)
                
                # Create results
                results = []
                for i in range(min(num_samples, len(true_labels))):
                    result = {
                        'true_label': int(true_labels[i].numpy()),
                        'predicted_label': int(predicted_labels[i]),
                        'prediction_confidence': float(np.max(predictions[i])),
                        'all_probabilities': predictions[i].tolist()
                    }
                    results.append(result)
                
                self.logger.info(f"Generated predictions for {len(results)} samples")
                return {'predictions': results}
                
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            return {'error': str(e)}
    
    def save_training_summary(self, data_info: Dict, training_result: Dict, 
                            test_results: Dict, predictions: Dict) -> str:
        """Save comprehensive training summary."""
        summary = {
            'experiment_info': {
                'timestamp': self.timestamp,
                'model_type': 'Hybrid Vision Transformer',
                'task': '4-class daily pattern classification',
                'dataset': 'NQ Futures Daily Patterns'
            },
            'data_info': data_info,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_result': training_result,
            'test_results': test_results,
            'sample_predictions': predictions,
            'class_weights': self.class_weights,
            'class_labels': self.config['classification']['labels']
        }
        
        # Add training history if available
        if self.training_history:
            summary['training_history'] = {
                'loss': self.training_history.history.get('loss', []),
                'accuracy': self.training_history.history.get('accuracy', []),
                'val_loss': self.training_history.history.get('val_loss', []),
                'val_accuracy': self.training_history.history.get('val_accuracy', [])
            }
        
        # Save summary
        summary_path = f"{self.paths['outputs']}/training_summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved: {summary_path}")
        return summary_path
    
    def run_complete_training(self, dataset_name: str = "test_daily_nq", 
                            epochs: int = None) -> Dict:
        """Run complete training pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING COMPLETE TRAINING PIPELINE")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info("=" * 60)
        
        try:
            # 1. Initialize components
            self.initialize_components()
            
            # 2. Prepare data
            data_info = self.prepare_data(dataset_name)
            
            # 3. Build model
            arch_path = self.build_model()
            
            # 4. Train model
            training_result = self.train_model(epochs)
            
            # 5. Evaluate model
            test_results = self.evaluate_model()
            
            # 6. Generate sample predictions
            predictions = self.generate_predictions()
            
            # 7. Save comprehensive summary
            summary_path = self.save_training_summary(
                data_info, training_result, test_results, predictions
            )
            
            self.logger.info("=" * 60)
            self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
            self.logger.info(f"Summary saved: {summary_path}")
            self.logger.info("=" * 60)
            
            return {
                'status': 'success',
                'summary_path': summary_path,
                'best_val_accuracy': self.best_val_accuracy,
                'timestamp': self.timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': self.timestamp
            }

def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description='Train daily NQ pattern analysis model')
    parser.add_argument('--dataset', type=str, default='test_daily_nq',
                       help='Dataset name to train on')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default from config)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    print("🚀 Starting Daily NQ Pattern Analysis Training")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    if args.epochs:
        print(f"Epochs: {args.epochs}")
    print()
    
    try:
        # Initialize trainer
        trainer = DailyModelTrainer(config_path=args.config)
        
        # Run complete training
        result = trainer.run_complete_training(
            dataset_name=args.dataset,
            epochs=args.epochs
        )
        
        if result['status'] == 'success':
            print("✅ Training completed successfully!")
            print(f"🎯 Best validation accuracy: {result['best_val_accuracy']:.4f}")
            print(f"📄 Summary: {result['summary_path']}")
            return 0
        else:
            print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())