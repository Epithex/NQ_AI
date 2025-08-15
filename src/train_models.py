#!/usr/bin/env python3
"""
Unified Training Interface for NQ_AI Models
Support for both Custom Hybrid ViT (3.49M) and ViT-Base Hybrid (87M) models
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main training interface with model selection."""
    parser = argparse.ArgumentParser(description='NQ_AI Model Training Interface')
    parser.add_argument('--model', type=str, choices=['hybrid', 'base'], default='hybrid',
                       help='Model type: hybrid (3.49M params) or base (87M params)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--test-only', action='store_true',
                       help='Test model creation without training')
    parser.add_argument('--compare', action='store_true',
                       help='Compare both model architectures')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 NQ_AI Multi-Model Training Interface")
    print("=" * 80)
    print(f"Selected model: {args.model}")
    print(f"Test mode: {args.test_only}")
    print(f"Comparison mode: {args.compare}")
    
    try:
        if args.compare:
            # Compare both models
            compare_models()
            return 0
        
        if args.model == 'hybrid':
            # Train custom hybrid ViT (3.49M params)
            print("\n🔧 Loading Custom Hybrid ViT (3.49M parameters)...")
            
            from train_daily_model import DailyModelTrainer
            config_path = args.config or 'config/config.yaml'
            
            print(f"Configuration: {config_path}")
            trainer = DailyModelTrainer(config_path)
            
            if args.test_only:
                print("🧪 Testing model creation...")
                trainer.build_model()
                print("✅ Custom Hybrid ViT test successful!")
                print(f"   Parameters: {trainer.model.count_params():,}")
            else:
                print("🏋️  Starting training...")
                results = trainer.run_complete_training(args.epochs)
                print("🎉 Custom Hybrid ViT training completed!")
                print(f"   Final accuracy: {results['test_accuracy']:.4f}")
        
        elif args.model == 'base':
            # Train ViT-Base hybrid (87M params)
            print("\n🔧 Loading ViT-Base Hybrid (87M parameters)...")
            
            from train_vit_base import ViTBaseTrainer
            config_path = args.config or 'config/config_vit_base.yaml'
            
            print(f"Configuration: {config_path}")
            trainer = ViTBaseTrainer(config_path)
            
            if args.test_only:
                print("🧪 Testing model creation...")
                trainer.create_model()
                print("✅ ViT-Base Hybrid test successful!")
                print(f"   Parameters: {trainer.model.count_params():,}")
            else:
                print("🏋️  Starting training...")
                results = trainer.run_complete_training(args.epochs)
                print("🎉 ViT-Base Hybrid training completed!")
                print(f"   Final accuracy: {results['test_accuracy']:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def compare_models():
    """Compare both model architectures."""
    print("\n📊 Comparing Model Architectures")
    print("=" * 50)
    
    try:
        # Import model classes
        from hybrid_vit_model import HybridViTModel
        from vit_base_hybrid_model import ViTBaseHybridModel
        
        # Test custom hybrid model
        print("🔧 Creating Custom Hybrid ViT...")
        hybrid_builder = HybridViTModel()
        hybrid_model = hybrid_builder.create_hybrid_model()
        hybrid_builder.compile_model()
        
        # Test ViT-Base hybrid model
        print("🔧 Creating ViT-Base Hybrid...")
        base_builder = ViTBaseHybridModel()
        base_model = base_builder.create_hybrid_model()
        base_builder.compile_model()
        
        # Compare models
        print("\n📈 Model Comparison Results:")
        print("-" * 50)
        
        hybrid_params = hybrid_model.count_params()
        base_params = base_model.count_params()
        
        print(f"Custom Hybrid ViT:")
        print(f"  • Parameters: {hybrid_params:,}")
        print(f"  • Size: ~{hybrid_params * 4 / 1e6:.1f} MB")
        print(f"  • Architecture: 6 transformer blocks, 256 projection dim")
        print(f"  • Training speed: Fast")
        print(f"  • Memory usage: Low")
        
        print(f"\nViT-Base Hybrid:")
        print(f"  • Parameters: {base_params:,}")
        print(f"  • Size: ~{base_params * 4 / 1e6:.1f} MB")
        print(f"  • Architecture: 12 transformer layers, 768 hidden size")
        print(f"  • Training speed: Slower")
        print(f"  • Memory usage: High")
        
        print(f"\nSize comparison:")
        ratio = base_params / hybrid_params
        print(f"  • ViT-Base is {ratio:.1f}x larger than Custom Hybrid")
        print(f"  • Memory difference: ~{(base_params - hybrid_params) * 4 / 1e6:.1f} MB")
        
        # Test inference speed with dummy data
        print("\n⚡ Testing Inference Speed:")
        print("-" * 30)
        
        import time
        batch_size = 4
        num_features = 3
        
        dummy_images = np.random.rand(batch_size, 224, 224, 3)
        dummy_features = np.random.rand(batch_size, num_features)
        
        # Warm up
        _ = hybrid_model.predict([dummy_images, dummy_features], verbose=0)
        _ = base_model.predict([dummy_images, dummy_features], verbose=0)
        
        # Time custom hybrid
        start_time = time.time()
        for _ in range(10):
            _ = hybrid_model.predict([dummy_images, dummy_features], verbose=0)
        hybrid_time = (time.time() - start_time) / 10
        
        # Time ViT-Base
        start_time = time.time()
        for _ in range(10):
            _ = base_model.predict([dummy_images, dummy_features], verbose=0)
        base_time = (time.time() - start_time) / 10
        
        print(f"Custom Hybrid ViT: {hybrid_time:.3f}s per batch")
        print(f"ViT-Base Hybrid:   {base_time:.3f}s per batch")
        print(f"Speed difference:   {base_time/hybrid_time:.1f}x slower for ViT-Base")
        
        print("\n💡 Recommendations:")
        print("-" * 20)
        print("• Use Custom Hybrid ViT for:")
        print("  - Fast experimentation")
        print("  - Limited computational resources")
        print("  - Real-time inference")
        print("• Use ViT-Base Hybrid for:")
        print("  - Maximum model capacity")
        print("  - Complex pattern recognition")
        print("  - Production systems with ample resources")
        
        print("\n✅ Model comparison completed successfully!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")
        raise

if __name__ == "__main__":
    exit(main())