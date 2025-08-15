#!/usr/bin/env python3
"""
NQ_AI Complete System Demonstration
Shows both model architectures and their capabilities
"""

import os
import sys

# Add src to path
sys.path.append('src')

def main():
    """Demonstrate the complete NQ_AI system."""
    print("=" * 80)
    print("🎯 NQ_AI Complete System Demonstration")
    print("=" * 80)
    print()
    
    print("📊 System Overview:")
    print("-" * 20)
    print("• Dataset: 6,235 daily NQ futures samples (2000-2025)")
    print("• Timeframe: Daily bars with 30-bar chart context")
    print("• Features: Chart images + 3 numerical features")
    print("• Classification: 4-class daily pattern recognition")
    print("• Models: Two hybrid architectures available")
    print()
    
    print("🏗️  Available Model Architectures:")
    print("-" * 35)
    print()
    
    print("1️⃣  Custom Hybrid ViT (3.49M parameters):")
    print("   • Architecture: 6 transformer blocks, 256 projection dim")
    print("   • Size: ~14 MB")
    print("   • Speed: Fast training and inference")
    print("   • Use case: Rapid experimentation, resource-constrained environments")
    print("   • Command: python src/train_models.py --model hybrid")
    print()
    
    print("2️⃣  ViT-Base Hybrid (87M parameters):")
    print("   • Architecture: Google ViT-Base-Patch16-224 + features")
    print("   • Size: ~347 MB (24.8x larger)")
    print("   • Speed: Slower but higher capacity")
    print("   • Use case: Maximum performance, production systems")
    print("   • Command: python src/train_models.py --model base")
    print()
    
    print("📋 Pattern Classification System:")
    print("-" * 32)
    print("• Label 1: High Breakout (daily high >= previous day high only)")
    print("• Label 2: Low Breakdown (daily low <= previous day low only)")
    print("• Label 3: Range Expansion (both levels touched)")
    print("• Label 4: Range Bound (neither level touched)")
    print()
    
    print("🔧 Quick Start Commands:")
    print("-" * 23)
    print("# Compare both models")
    print("python src/train_models.py --compare")
    print()
    print("# Test model creation")
    print("python src/train_models.py --model hybrid --test-only")
    print("python src/train_models.py --model base --test-only")
    print()
    print("# Train models")
    print("python src/train_models.py --model hybrid --epochs 20")
    print("python src/train_models.py --model base --epochs 15")
    print()
    print("# Individual training scripts")
    print("python src/train_daily_model.py")
    print("python src/train_vit_base.py")
    print()
    
    print("📁 Key Files Created:")
    print("-" * 20)
    print("Configuration:")
    print("  • config/config.yaml - Custom hybrid ViT settings")
    print("  • config/config_vit_base.yaml - ViT-Base hybrid settings")
    print()
    print("Model Implementation:")
    print("  • src/hybrid_vit_model.py - Custom 3.49M parameter model")
    print("  • src/vit_base_hybrid_model.py - Google ViT-Base 87M parameter model")
    print()
    print("Training Scripts:")
    print("  • src/train_daily_model.py - Custom hybrid ViT trainer")
    print("  • src/train_vit_base.py - ViT-Base hybrid trainer")
    print("  • src/train_models.py - Unified training interface")
    print()
    print("Data Pipeline:")
    print("  • src/daily_dataset_creator.py - Generate 25-year dataset")
    print("  • src/daily_data_loader.py - Load data for training")
    print()
    
    print("🎯 System Status:")
    print("-" * 16)
    print("✅ Dataset Generation: 6,235 samples ready")
    print("✅ Custom Hybrid ViT: 3.49M params, fully operational")  
    print("✅ ViT-Base Hybrid: 87M params, fully operational")
    print("✅ Training Interface: Unified model selection")
    print("✅ Documentation: Updated for daily-only system")
    print("❌ Hourly System: Completely removed as requested")
    print()
    
    print("📈 Next Steps:")
    print("-" * 13)
    print("1. Choose your model architecture based on requirements")
    print("2. Run training with your preferred configuration")
    print("3. Compare results between both model architectures")
    print("4. Deploy the best-performing model for your use case")
    print()
    
    print("=" * 80)
    print("🚀 Ready to train! Use the commands above to get started.")
    print("=" * 80)

if __name__ == "__main__":
    main()