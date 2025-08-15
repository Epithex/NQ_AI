#!/usr/bin/env python3
"""
Simple ViT Training Script for NQ_AI
Quick training entry point with basic configuration
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add models directory to path
sys.path.append('models')

from models.data_loader import create_data_loaders
from models.vit_trainer import NQViTTrainer
from models.evaluation import NQModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function with basic configuration."""
    logger.info("=" * 60)
    logger.info("NQ_AI Vision Transformer Training")
    logger.info("=" * 60)
    
    # Basic configuration
    config = {
        'batch_size': 8,  # Smaller batch size for stability
        'learning_rate': 5e-5,  # Conservative learning rate
        'epochs': 30,  # Reasonable number for initial training
        'image_size': (224, 224)
    }
    
    logger.info(f"Training configuration: {config}")
    
    try:
        # Create data loaders
        logger.info("Loading datasets...")
        train_dataset, test_dataset, data_info = create_data_loaders(
            batch_size=config['batch_size'],
            image_size=config['image_size']
        )
        
        logger.info(f"Train samples: {data_info['train']['num_samples']}")
        logger.info(f"Test samples: {data_info['test']['num_samples']}")
        logger.info(f"Label distribution: {data_info['train']['label_distribution']}")
        
        # Initialize trainer
        logger.info("Initializing ViT trainer...")
        trainer = NQViTTrainer(
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs']
        )
        
        # Train model
        logger.info("Starting training...")
        model = trainer.train(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            class_weights=data_info['class_weights']
        )
        
        # Evaluate model
        logger.info("Evaluating trained model...")
        results = trainer.evaluate(test_dataset)
        
        # Save results
        trainer.save_results(results)
        
        # Generate evaluation report
        evaluator = NQModelEvaluator(results_dir="models/outputs")
        detailed_results = evaluator.evaluate_model(test_dataset)
        evaluator.save_evaluation_results(detailed_results)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Final test accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Model saved to: models/outputs/checkpoints/best_model.h5")
        logger.info(f"Detailed results saved to: models/outputs/")
        logger.info(f"TensorBoard logs: models/outputs/logs/")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Full error:", exc_info=True)
        return None

if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n🎯 Training Summary:")
        print(f"   Accuracy: {results['test_accuracy']:.1%}")
        print(f"   Loss: {results['test_loss']:.4f}")
        print(f"   Files: models/outputs/")
        print(f"\n✅ Ready for NQ futures trading predictions!")
    else:
        print(f"\n❌ Training failed. Check logs above.")
        sys.exit(1)