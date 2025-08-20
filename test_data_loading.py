#!/usr/bin/env python3
"""
Test script to verify pure visual data loading with actual generated dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from pure_visual_data_loader import PureVisualDataLoader
import tensorflow as tf

def main():
    print("🧪 Testing Pure Visual Data Loading with Real Dataset")
    print("=" * 60)
    
    try:
        # Initialize data loader
        data_loader = PureVisualDataLoader("config/config_pure_visual_daily.yaml")
        print("✅ Data loader initialized successfully")
        
        # Test loading dataset manifest
        print("\n📊 Testing dataset manifest loading...")
        manifest = data_loader.load_dataset_manifest()
        print(f"✅ Manifest loaded: {len(manifest.get('samples', []))} samples")
        print(f"   Model type: {manifest.get('model_type')}")
        print(f"   Instruments: {list(manifest.get('instruments', {}).keys())}")
        
        # Test sample index loading
        print("\n📋 Testing sample index loading...")
        sample_index = data_loader.load_sample_index()
        print(f"✅ Sample index loaded: {len(sample_index)} samples")
        
        # Check sample structure
        if sample_index:
            sample = sample_index[0]
            print(f"   Sample structure: {list(sample.keys())}")
            print(f"   Pure visual: {sample.get('pure_visual', False)}")
            print(f"   Has numerical features: {sample.get('numerical_features') is not None}")
            print(f"   Chart path exists: {Path(sample['chart_path']).exists()}")
        
        # Test data splits
        print("\n📊 Testing data splits...")
        train_samples, val_samples, test_samples = data_loader.create_data_splits(sample_index)
        print(f"✅ Data splits created:")
        print(f"   Train: {len(train_samples)} samples")
        print(f"   Validation: {len(val_samples)} samples") 
        print(f"   Test: {len(test_samples)} samples")
        
        # Test class weights calculation
        print("\n⚖️ Testing class weights...")
        class_weights = data_loader.calculate_class_weights(train_samples)
        print(f"✅ Class weights calculated: {class_weights}")
        
        # Test TensorFlow dataset creation
        print("\n🔧 Testing TensorFlow dataset creation...")
        train_dataset = data_loader.create_tf_dataset(
            train_samples[:10],  # Use small sample for testing
            batch_size=2, 
            shuffle=False, 
            repeat=False
        )
        print("✅ TensorFlow dataset created")
        
        # Test dataset validation
        print("\n🧪 Testing dataset validation...")
        is_valid = data_loader.validate_dataset(train_dataset, num_batches=2)
        print(f"✅ Dataset validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test actual data loading
        print("\n📈 Testing actual batch loading...")
        for i, (images, labels) in enumerate(train_dataset.take(1)):
            print(f"✅ Loaded batch {i+1}:")
            print(f"   Images shape: {images.shape}")
            print(f"   Images dtype: {images.dtype}")
            print(f"   Images range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Labels dtype: {labels.dtype}")
            print(f"   Labels range: [{tf.reduce_min(labels)}, {tf.reduce_max(labels)}]")
            print(f"   Label values: {labels.numpy()}")
        
        print("\n✅ Pure Visual Data Loading Test COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for model creation and training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)