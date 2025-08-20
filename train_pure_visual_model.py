#!/usr/bin/env python3
"""
Train Pure Visual ViT Model for "The Test"
Tests if ViT can learn patterns from 448x448 charts without numerical crutch
Success: >40% accuracy proves visual learning viability
"""

import sys
import os
sys.path.append("src")

import argparse
import logging
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Import pure visual components
from pure_visual_vit_model import PureVisualViTModel
from daily_data_loader import DailyDataLoader


def parse_arguments():
    """Parse command line arguments for pure visual training."""
    parser = argparse.ArgumentParser(
        description="Train Pure Visual ViT Model for 'The Test'"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_pure_visual_daily.yaml",
        help="Pure visual configuration file",
    )
    
    parser.add_argument(
        "--dataset_manifest",
        type=str,
        default=None,
        help="Path to pure visual dataset manifest (auto-detect if None)",
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only run test evaluation (requires trained model)",
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions for analysis",
    )
    
    return parser.parse_args()


def setup_logging():
    """Setup logging for pure visual training."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/pure_visual_training.log"),
            logging.StreamHandler(),
        ],
    )
    
    return logging.getLogger(__name__)


def find_latest_dataset_manifest(metadata_dir: str) -> str:
    """Find the most recent pure visual dataset manifest."""
    metadata_path = Path(metadata_dir)
    
    # Look for pure visual manifests
    manifests = list(metadata_path.glob("*pure_visual*manifest*.json"))
    
    if not manifests:
        # Fallback to daily manifests
        manifests = list(metadata_path.glob("*daily*manifest*.json"))
    
    if not manifests:
        raise FileNotFoundError(f"No dataset manifest found in {metadata_dir}")
    
    # Return the most recent
    latest_manifest = max(manifests, key=lambda x: x.stat().st_mtime)
    return str(latest_manifest)


def load_dataset_info(manifest_path: str, logger) -> dict:
    """Load dataset information from manifest."""
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        total_samples = len(manifest.get("samples", []))
        
        # Check if this is a pure visual dataset
        sample = manifest["samples"][0] if manifest.get("samples") else {}
        has_numerical = "numerical_features" in sample
        
        dataset_info = {
            "manifest": manifest,
            "total_samples": total_samples,
            "is_pure_visual": not has_numerical,
            "instruments": manifest.get("instruments", ["UNKNOWN"]),
            "date_range": f"{manifest.get('start_date', 'N/A')} to {manifest.get('end_date', 'N/A')}",
            "image_size": manifest.get("image_size", 224),
        }
        
        logger.info(f"📊 Dataset Info:")
        logger.info(f"   Total samples: {total_samples:,}")
        logger.info(f"   Pure visual: {dataset_info['is_pure_visual']}")
        logger.info(f"   Instruments: {dataset_info['instruments']}")
        logger.info(f"   Date range: {dataset_info['date_range']}")
        logger.info(f"   Image size: {dataset_info['image_size']}x{dataset_info['image_size']}")
        
        return dataset_info
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset manifest: {e}")
        raise


def evaluate_the_test_results(accuracy: float, logger) -> str:
    """Evaluate results of 'The Test' based on accuracy."""
    if accuracy >= 0.40:  # 40% threshold
        result = "SUCCESS"
        interpretation = "Visual learning is VIABLE - ViT can learn patterns from charts"
        recommendation = "Continue pure visual development with optimizations"
    elif accuracy >= 0.30:  # 30-40% gray area
        result = "MIXED"
        interpretation = "Visual learning shows promise but needs improvement"
        recommendation = "Try enhanced visual features or hybrid approach"
    else:  # <30% failure
        result = "FAILURE"
        interpretation = "Numerical features are SUPERIOR - visual learning insufficient"
        recommendation = "Pivot to pure numerical approach with better feature engineering"
    
    logger.info("🔬 'THE TEST' RESULTS:")
    logger.info(f"   Final accuracy: {accuracy:.1%}")
    logger.info(f"   Result: {result}")
    logger.info(f"   Interpretation: {interpretation}")
    logger.info(f"   Recommendation: {recommendation}")
    
    return result


def main():
    """Main function for pure visual training."""
    print("🚀 NQ_AI Pure Visual ViT Training - 'The Test'")
    print("Testing visual learning without numerical crutch")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Starting PURE VISUAL ViT training for 'The Test'")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Load configuration
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify pure visual setup
        if config.get("features", {}).get("numerical_features", 3) != 0:
            logger.error("❌ Config error: This is not a pure visual configuration!")
            return 1
        
        logger.info("✅ Pure visual configuration confirmed")
        
        # Find dataset manifest
        if args.dataset_manifest:
            manifest_path = args.dataset_manifest
        else:
            metadata_dir = config["paths"]["metadata"]
            manifest_path = find_latest_dataset_manifest(metadata_dir)
        
        logger.info(f"📁 Using dataset manifest: {manifest_path}")
        
        # Load dataset information
        dataset_info = load_dataset_info(manifest_path, logger)
        
        if not dataset_info["is_pure_visual"]:
            logger.warning("⚠️  Dataset contains numerical features - not pure visual!")
        
        # Initialize pure visual model
        logger.info("🏗️  Initializing Pure Visual ViT model...")
        model_builder = PureVisualViTModel(config_path=args.config)
        
        # Create model
        logger.info("🎯 Creating pure visual model...")
        model = model_builder.create_model()
        
        # Compile model
        logger.info("⚙️  Compiling model...")
        model_builder.compile_model()
        
        # Print model summary
        total_params = model.count_params()
        logger.info(f"🧠 Model Summary:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   ALL parameters dedicated to visual learning")
        logger.info(f"   Input: {dataset_info['image_size']}x{dataset_info['image_size']}x3 (pure visual)")
        logger.info(f"   Output: 4 classes (previous day levels)")
        logger.info(f"   Architecture: ViT-Base (87M+ params)")
        
        # Load data
        logger.info("📊 Loading pure visual dataset...")
        data_loader = DailyDataLoader(config_path=args.config)
        
        # For pure visual, we need to modify the data loader to skip numerical features
        train_dataset, val_dataset, test_dataset = data_loader.load_datasets(
            dataset_manifest_path=manifest_path,
            pure_visual=True  # Force pure visual mode
        )
        
        logger.info(f"📈 Dataset splits loaded for pure visual training")
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"🔄 Resuming from checkpoint: {args.resume_from}")
            model.load_weights(args.resume_from)
        
        # Test only mode
        if args.test_only:
            if not args.resume_from:
                logger.error("❌ Test-only mode requires --resume_from checkpoint")
                return 1
            
            logger.info("🧪 Running test-only evaluation...")
            test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
            
            logger.info(f"🎯 Test Results:")
            logger.info(f"   Test Loss: {test_loss:.4f}")
            logger.info(f"   Test Accuracy: {test_accuracy:.1%}")
            
            # Evaluate 'The Test' results
            evaluate_the_test_results(test_accuracy, logger)
            
            return 0
        
        # Setup training callbacks
        callbacks = []
        
        # Model checkpointing
        checkpoint_dir = config["paths"]["checkpoints"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{checkpoint_dir}/pure_visual_vit_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping for 'The Test' (shorter patience)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # TensorBoard logging
        tensorboard_dir = config["paths"]["tensorboard_logs"]
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=f"{tensorboard_dir}/pure_visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            histogram_freq=1,
            write_graph=True,
            write_images=False
        )
        callbacks.append(tensorboard_callback)
        
        # Start training
        epochs = config["model"]["epochs"]
        
        logger.info("🚀 Starting PURE VISUAL training for 'The Test'...")
        logger.info(f"   Target epochs: {epochs}")
        logger.info(f"   Success criteria: >40% validation accuracy")
        logger.info(f"   Failure criteria: <30% validation accuracy")
        
        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best validation accuracy
        best_val_accuracy = max(history.history['val_accuracy'])
        
        logger.info(f"🏁 Training completed!")
        logger.info(f"   Best validation accuracy: {best_val_accuracy:.1%}")
        
        # Final test evaluation
        logger.info("🧪 Running final test evaluation...")
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
        
        logger.info(f"🎯 Final Test Results:")
        logger.info(f"   Test Loss: {test_loss:.4f}")
        logger.info(f"   Test Accuracy: {test_accuracy:.1%}")
        
        # Evaluate 'The Test' results
        test_result = evaluate_the_test_results(test_accuracy, logger)
        
        # Save final results
        results = {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "best_val_accuracy": float(best_val_accuracy),
            "total_params": int(total_params),
            "dataset_samples": dataset_info["total_samples"],
            "image_size": dataset_info["image_size"],
            "the_test_result": test_result,
            "timestamp": datetime.now().isoformat(),
            "config_used": args.config
        }
        
        # Save results
        results_file = f"{config['paths']['outputs']}/pure_visual_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Results saved: {results_file}")
        
        # Final summary
        print(f"\n🔬 'THE TEST' FINAL SUMMARY:")
        print(f"   Pure Visual ViT Accuracy: {test_accuracy:.1%}")
        print(f"   Numerical Baseline: 61% (from previous test)")
        print(f"   Result: {test_result}")
        
        if test_result == "SUCCESS":
            print(f"✅ CONCLUSION: Visual learning is viable! Continue ViT development.")
        elif test_result == "MIXED":
            print(f"⚠️  CONCLUSION: Visual learning shows promise but needs optimization.")
        else:
            print(f"❌ CONCLUSION: Numerical features superior. Pivot to numerical approach.")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())