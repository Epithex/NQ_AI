#!/usr/bin/env python3
"""
Test numerical-only model performance vs hybrid model
Critical test to see if visual branch is being ignored
"""

import sys
import os
sys.path.append("src")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json

def load_numerical_features_and_labels():
    """Load numerical features and labels from the dataset."""
    
    # Load the dataset manifest
    manifest_path = "data/metadata_daily/daily_hybrid_dataset_manifest_20250819_210117.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Extract numerical features and labels from samples
    features = []
    labels = []
    
    sample_index = manifest["samples"]
    for sample in sample_index[:1000]:  # Test with first 1000 samples for speed
        # Numerical features: [distance_to_prev_high, distance_to_prev_low, prev_day_range]
        numerical_features = sample["numerical_features"]
        label = sample["pattern"] - 1  # Convert 1-4 to 0-3
        
        features.append(numerical_features)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def test_numerical_only_models():
    """Test different models using only numerical features."""
    
    print("🧪 Testing Numerical-Only Models")
    print("="*50)
    
    # Load data
    X, y = load_numerical_features_and_labels()
    print(f"Loaded {len(X)} samples with {X.shape[1]} numerical features")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Split data (80/20 train/test to match our training setup)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Test multiple models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        print("   Classification Report:")
        report = classification_report(y_test, y_pred, 
                                     target_names=['High Breakout', 'Low Breakdown', 'Range Expansion', 'Range Bound'],
                                     digits=4)
        print("   " + "\n   ".join(report.split("\n")))
    
    # Summary
    print(f"\n📊 RESULTS SUMMARY")
    print("="*50)
    for name, accuracy in results.items():
        print(f"{name:20s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Compare to hybrid model result
    hybrid_accuracy = 0.611  # Our best result
    print(f"\nHybrid ViT Model     : {hybrid_accuracy:.4f} ({hybrid_accuracy*100:.2f}%)")
    
    # Analysis
    best_numerical = max(results.values())
    print(f"\n🔍 ANALYSIS:")
    print(f"   Best numerical-only accuracy: {best_numerical:.4f}")
    print(f"   Hybrid model accuracy:        {hybrid_accuracy:.4f}")
    print(f"   Difference:                   {hybrid_accuracy - best_numerical:.4f}")
    
    if abs(hybrid_accuracy - best_numerical) < 0.05:  # Within 5%
        print(f"   ⚠️  CONCLUSION: Visual branch likely IGNORED!")
        print(f"       The hybrid model performs similarly to numerical-only models.")
        print(f"       The 86.6M parameter ViT is probably not contributing meaningful information.")
    else:
        print(f"   ✅ CONCLUSION: Visual branch IS contributing!")
        print(f"       Significant performance difference suggests visual information is being used.")
    
    return results

if __name__ == "__main__":
    test_numerical_only_models()