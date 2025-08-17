#!/usr/bin/env python3
"""
Quick analysis of the generated NQ_AI dataset
"""

import json
import os
from collections import defaultdict


def analyze_dataset():
    labels_dir = "data/labels"

    # Count labels
    label_counts = defaultdict(int)
    total_samples = 0

    for filename in os.listdir(labels_dir):
        if filename.endswith(".json"):
            with open(os.path.join(labels_dir, filename), "r") as f:
                data = json.load(f)
                label = data["actual_label"]
                label_counts[label] += 1
                total_samples += 1

    print("=" * 50)
    print("NQ_AI Dataset Analysis")
    print("=" * 50)
    print(f"Total Samples Generated: {total_samples}")
    print()
    print("Label Distribution:")
    print("-" * 30)

    label_names = {
        1: "Only green line (prev day high)",
        2: "Only red line (prev day low)",
        3: "Green first, then red",
        4: "Red first, then green",
        5: "Rangebound (neither line)",
        6: "Simultaneous touches",
    }

    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / total_samples) * 100
        print(
            f"Label {label}: {count:3d} samples ({percentage:5.1f}%) - {label_names[label]}"
        )

    print()
    print("Dataset Features:")
    print("-" * 30)
    print("✓ 300-bar charts with color-coded previous day levels")
    print("✓ 6-class classification system")
    print("✓ Priority-based classification (5→6→1-4)")
    print("✓ Success/failure matrix for each sample")
    print("✓ Detailed touch information with timestamps")
    print("✓ High-resolution chart images for AI training")
    print()
    print("Files Generated:")
    print("-" * 30)
    print(
        f"Chart Images: {len([f for f in os.listdir('data/images') if f.endswith('.png')])} PNG files"
    )
    print(f"Label Files: {total_samples} JSON files")
    print(f"Metadata: {len(os.listdir('data/metadata'))} summary files")


if __name__ == "__main__":
    analyze_dataset()
