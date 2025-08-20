#!/usr/bin/env python3
"""
Test script to verify pure visual ViT model creation and compilation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from pure_visual_daily_vit_model import PureVisualDailyViTModel
import tensorflow as tf

def main():
    print("🧪 Testing Pure Visual ViT Model Creation and Compilation")
    print("=" * 65)
    
    try:
        # Initialize model builder
        print("🏗️ Initializing pure visual ViT model builder...")
        model_builder = PureVisualDailyViTModel("config/config_pure_visual_daily.yaml")
        print("✅ Model builder initialized successfully")
        
        # Test model creation
        print("\n🔧 Testing model creation...")
        model = model_builder.create_model()
        print("✅ Model created successfully")
        
        # Display model information
        print(f"\n📊 Model Information:")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        print(f"   Model inputs: {len(model.inputs)}")
        print(f"   Model outputs: {len(model.outputs)}")
        
        # Check input shape
        if model.inputs:
            input_shape = model.inputs[0].shape
            print(f"   Input shape: {input_shape}")
            
        # Check output shape  
        if model.outputs:
            output_shape = model.outputs[0].shape
            print(f"   Output shape: {output_shape}")
        
        # Test model compilation
        print("\n⚙️ Testing model compilation...")
        
        # Test without class weights first
        model_builder.compile_model(class_weights=None)
        print("✅ Model compiled successfully (no class weights)")
        
        # Test with class weights
        test_class_weights = {0: 1.0, 1: 1.2, 2: 1.8, 3: 2.1}
        model_builder.compile_model(class_weights=test_class_weights)
        print("✅ Model compiled successfully (with class weights)")
        
        # Test model summary
        print("\n📋 Model Summary:")
        try:
            model.summary(line_length=80)
            print("✅ Model summary generated successfully")
        except Exception as e:
            print(f"⚠️ Model summary failed: {e}")
        
        # Test forward pass with dummy data
        print("\n🚀 Testing forward pass...")
        dummy_input = tf.random.normal((2, 448, 448, 3))  # Batch of 2 images
        print(f"   Dummy input shape: {dummy_input.shape}")
        
        outputs = model(dummy_input, training=False)
        print(f"   Model output shape: {outputs.shape}")
        print(f"   Output range: [{tf.reduce_min(outputs):.3f}, {tf.reduce_max(outputs):.3f}]")
        print("✅ Forward pass successful")
        
        # Test prediction probabilities
        print("\n🎯 Testing prediction probabilities...")
        predictions = tf.nn.softmax(outputs)
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Prediction sum per sample: {tf.reduce_sum(predictions, axis=1)}")
        print(f"   Sample predictions: {predictions[0].numpy()}")
        print("✅ Predictions look correct (sum to 1.0)")
        
        # Test memory usage
        print("\n💾 Testing memory usage...")
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"   Current memory usage: {memory_mb:.1f} MB")
        except ImportError:
            print("   Memory monitoring not available (psutil not installed)")
        
        print("\n✅ Pure Visual ViT Model Creation Test COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for end-to-end training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model creation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)