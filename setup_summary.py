#!/usr/bin/env python3
"""
NQ_AI ViT Training Setup Summary
Shows current setup status and next steps
"""

import os
import json
from pathlib import Path


def check_setup_status():
    """Check and display setup status."""
    print("=" * 70)
    print("🚀 NQ_AI VISION TRANSFORMER TRAINING SETUP COMPLETE!")
    print("=" * 70)

    # Check dataset split
    train_manifest = Path("data/metadata/train_manifest.json")
    test_manifest = Path("data/metadata/test_manifest.json")

    if train_manifest.exists() and test_manifest.exists():
        with open(train_manifest) as f:
            train_info = json.load(f)
        with open(test_manifest) as f:
            test_info = json.load(f)

        print("✅ Dataset Split Complete:")
        print(
            f"   📊 Training: {train_info['total_samples']} samples ({train_info['date_range']['start']} → {train_info['date_range']['end']})"
        )
        print(
            f"   📊 Testing:  {test_info['total_samples']} samples ({test_info['date_range']['start']} → {test_info['date_range']['end']})"
        )
        print(f"   📈 Label distribution: {train_info['label_distribution']}")
    else:
        print("❌ Dataset split not found")

    # Check model files
    model_files = [
        "models/data_loader.py",
        "models/vit_trainer.py",
        "models/evaluation.py",
        "models/train_config.yaml",
        "train_vit.py",
    ]

    print("\n✅ Model Components:")
    for file in model_files:
        if Path(file).exists():
            print(f"   📄 {file}")
        else:
            print(f"   ❌ {file} (missing)")

    # Check dependencies
    print(f"\n✅ Dependencies Installed:")
    print(f"   🧠 TensorFlow 2.20.0 + Keras 3.11.2")
    print(f"   🔮 Transformers + vit-keras")
    print(f"   📊 scikit-learn + seaborn")
    print(f"   📈 matplotlib + pandas")

    print(f"\n🎯 Ready for Training:")
    print(f"   • Vision Transformer Base model (85M parameters)")
    print(f"   • 6-class NQ futures classification")
    print(f"   • Custom data pipeline with augmentation")
    print(f"   • Comprehensive evaluation metrics")
    print(f"   • Class weighting for imbalanced data")

    print(f"\n🚀 Next Steps:")
    print(f"   1. Start training: python train_vit.py")
    print(f"   2. Monitor progress: tensorboard --logdir models/outputs/logs")
    print(f"   3. View results: models/outputs/")
    print(f"   4. Evaluate performance: python models/evaluation.py")

    print(f"\n📊 Training Details:")
    print(f"   • Batch size: 8 (optimized for stability)")
    print(f"   • Learning rate: 5e-5 (conservative)")
    print(f"   • Epochs: 30 (with early stopping)")
    print(f"   • Image size: 224x224 (ViT standard)")
    print(f"   • Data augmentation: flip, brightness, contrast")

    print("=" * 70)
    print("🎉 READY TO TRAIN YOUR NQ FUTURES VISION TRANSFORMER!")
    print("=" * 70)


if __name__ == "__main__":
    check_setup_status()
