#!/usr/bin/env python3
"""
Test script to validate pure visual training setup and configuration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from train_pure_visual_daily import PureVisualDailyTrainer
import yaml

def test_configuration():
    """Test configuration validation."""
    print("🧪 Testing Configuration Validation")
    print("=" * 50)
    
    try:
        # Test with main config
        print("📋 Testing main configuration...")
        trainer = PureVisualDailyTrainer("config/config_pure_visual_daily.yaml")
        print("✅ Main configuration validation passed")
        
        # Check specific parameters
        config = trainer.config
        model_config = config['model']
        training_config = config['training']
        
        print(f"\n📊 Configuration Summary:")
        print(f"   Batch Size: {model_config['batch_size']}")
        print(f"   Learning Rate: {model_config['learning_rate']}")
        print(f"   Epochs: {model_config['epochs']}")
        print(f"   Weight Decay: {model_config['weight_decay']}")
        print(f"   Optimizer: {training_config.get('optimizer', 'adamw')}")
        print(f"   LR Schedule: {training_config.get('lr_schedule', 'cosine_annealing')}")
        print(f"   Early Stopping: {training_config.get('early_stopping', False)}")
        
        # Validate scheduler parameters
        if training_config.get('lr_schedule') == 'cosine_annealing':
            t_max = training_config.get('lr_scheduler_t_max', model_config['epochs'])
            min_lr = training_config.get('lr_scheduler_min_lr', 1e-5)
            print(f"   Scheduler T_max: {t_max}")
            print(f"   Scheduler min_lr: {min_lr}")
            
            # Verify scheduler math
            import math
            initial_lr = model_config['learning_rate']
            test_epoch = 200  # Middle of training
            expected_lr = min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * test_epoch / t_max)) / 2
            print(f"   LR at epoch {test_epoch}: {expected_lr:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_training_components():
    """Test training components initialization."""
    print("\n🧪 Testing Training Components")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = PureVisualDailyTrainer("config/config_pure_visual_daily.yaml")
        
        # Test callback creation
        print("📋 Testing callback creation...")
        callbacks = trainer.create_callbacks()
        
        callback_types = [type(cb).__name__ for cb in callbacks]
        print(f"✅ Created {len(callbacks)} callbacks: {callback_types}")
        
        # Verify only one scheduler is active
        scheduler_callbacks = [cb for cb in callbacks if 'LearningRate' in type(cb).__name__]
        print(f"   Learning rate schedulers: {len(scheduler_callbacks)}")
        
        if len(scheduler_callbacks) > 1:
            print("⚠️  WARNING: Multiple learning rate schedulers detected!")
            return False
        elif len(scheduler_callbacks) == 1:
            print("✅ Single learning rate scheduler confirmed")
        else:
            print("ℹ️  No learning rate scheduler (may be disabled)")
        
        # Check early stopping
        early_stopping_callbacks = [cb for cb in callbacks if 'EarlyStopping' in type(cb).__name__]
        if trainer.config['training'].get('early_stopping', False):
            if len(early_stopping_callbacks) == 1:
                print("✅ Early stopping callback configured")
            else:
                print("⚠️  Early stopping enabled but callback missing")
        else:
            print("ℹ️  Early stopping disabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation with new parameters."""
    print("\n🧪 Testing Model Creation")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = PureVisualDailyTrainer("config/config_pure_visual_daily.yaml")
        
        # Build model
        print("🏗️ Building model...")
        trainer.build_model()
        
        if trainer.model is not None:
            print(f"✅ Model created successfully")
            print(f"   Parameters: {trainer.model.count_params():,}")
            print(f"   Input shape: {trainer.model.input.shape}")
            print(f"   Output shape: {trainer.model.output.shape}")
        else:
            print("❌ Model creation failed")
            return False
        
        # Test optimizer configuration
        optimizer = trainer.model.optimizer
        print(f"   Optimizer type: {type(optimizer).__name__}")
        print(f"   Learning rate: {float(optimizer.learning_rate)}")
        
        # Handle mixed precision optimizer wrapper
        actual_optimizer = optimizer
        if hasattr(optimizer, '_optimizer'):  # Mixed precision wrapper
            actual_optimizer = optimizer._optimizer
            print(f"   Actual optimizer: {type(actual_optimizer).__name__}")
        
        if hasattr(actual_optimizer, 'weight_decay') and actual_optimizer.weight_decay is not None:
            print(f"   Weight decay: {float(actual_optimizer.weight_decay)}")
        else:
            # Weight decay is in the config but may not be accessible due to mixed precision
            weight_decay = trainer.config['model'].get('weight_decay', 'Not specified')
            print(f"   Weight decay: {weight_decay} (configured, may be wrapped by mixed precision)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 NQ_AI Pure Visual Training Setup Validation")
    print("=" * 60)
    
    tests = [
        test_configuration,
        test_training_components,
        test_model_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results Summary:")
    print(f"=" * 30)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test.__name__}: {status}")
    
    all_passed = all(results)
    if all_passed:
        print(f"\n🎉 All tests PASSED! Training setup is ready.")
        print(f"🚀 Safe to proceed with:")
        print(f"   python src/train_pure_visual_daily.py --config config/config_pure_visual_daily.yaml")
    else:
        print(f"\n⚠️  Some tests FAILED! Please review configuration.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())