#!/usr/bin/env python3
"""
Dataset Splitter for NQ_AI ViT Training
Creates 80:20 train/test split maintaining temporal order
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import yaml
from collections import Counter

class DatasetSplitter:
    """Split NQ_AI dataset into train/test sets for ViT training."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DatasetSplitter with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.source_images = Path("data/images")
        self.source_labels = Path("data/labels") 
        self.train_split = 0.8
        
        # Create target directories
        self.train_dir = Path("data/train")
        self.test_dir = Path("data/test")
        
        for split_dir in [self.train_dir, self.test_dir]:
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    def get_all_samples(self):
        """Get all samples sorted by date for temporal split."""
        samples = []
        
        for label_file in self.source_labels.glob("*.json"):
            with open(label_file, 'r') as f:
                data = json.load(f)
            
            date_str = data['date']
            chart_image = data['chart_image']
            
            # Verify image exists
            if Path(chart_image).exists():
                samples.append({
                    'date': date_str,
                    'label_file': label_file,
                    'image_file': Path(chart_image),
                    'label': data['actual_label']
                })
        
        # Sort by date to maintain temporal order
        samples.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
        
        print(f"Found {len(samples)} valid samples")
        print(f"Date range: {samples[0]['date']} to {samples[-1]['date']}")
        
        return samples
    
    def split_samples(self, samples):
        """Split samples into train/test maintaining temporal order."""
        total_samples = len(samples)
        train_size = int(total_samples * self.train_split)
        
        train_samples = samples[:train_size]
        test_samples = samples[train_size:]
        
        print(f"\nDataset Split:")
        print(f"Training: {len(train_samples)} samples ({len(train_samples)/total_samples*100:.1f}%)")
        print(f"Testing:  {len(test_samples)} samples ({len(test_samples)/total_samples*100:.1f}%)")
        print(f"Train date range: {train_samples[0]['date']} to {train_samples[-1]['date']}")
        print(f"Test date range:  {test_samples[0]['date']} to {test_samples[-1]['date']}")
        
        return train_samples, test_samples
    
    def copy_files(self, samples, target_dir, split_name):
        """Copy files to target directory and create manifest."""
        print(f"\nCopying {split_name} files...")
        
        manifest = {
            'split': split_name,
            'total_samples': len(samples),
            'date_range': {
                'start': samples[0]['date'],
                'end': samples[-1]['date']
            },
            'label_distribution': {},
            'samples': []
        }
        
        # Count label distribution
        label_counts = Counter(sample['label'] for sample in samples)
        manifest['label_distribution'] = dict(label_counts)
        
        for i, sample in enumerate(samples):
            # Copy image file
            target_image = target_dir / "images" / sample['image_file'].name
            shutil.copy2(sample['image_file'], target_image)
            
            # Copy label file
            target_label = target_dir / "labels" / sample['label_file'].name
            shutil.copy2(sample['label_file'], target_label)
            
            # Update manifest
            manifest['samples'].append({
                'date': sample['date'],
                'image': str(target_image.relative_to(Path("data"))),
                'label_file': str(target_label.relative_to(Path("data"))),
                'label': sample['label']
            })
            
            if (i + 1) % 50 == 0:
                print(f"  Copied {i + 1}/{len(samples)} files...")
        
        # Save manifest
        manifest_file = Path("data/metadata") / f"{split_name}_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"  Completed {split_name}: {len(samples)} files")
        print(f"  Manifest saved: {manifest_file}")
        
        return manifest
    
    def create_split(self):
        """Create complete train/test split."""
        print("=" * 60)
        print("NQ_AI Dataset Splitter for ViT Training")
        print("=" * 60)
        
        # Get all samples
        samples = self.get_all_samples()
        
        # Split samples
        train_samples, test_samples = self.split_samples(samples)
        
        # Copy files and create manifests
        train_manifest = self.copy_files(train_samples, self.train_dir, "train")
        test_manifest = self.copy_files(test_samples, self.test_dir, "test")
        
        # Create summary
        summary = {
            'split_date': datetime.now().isoformat(),
            'train_split': self.train_split,
            'total_samples': len(samples),
            'train': {
                'samples': len(train_samples),
                'percentage': len(train_samples) / len(samples) * 100,
                'date_range': train_manifest['date_range'],
                'label_distribution': train_manifest['label_distribution']
            },
            'test': {
                'samples': len(test_samples),
                'percentage': len(test_samples) / len(samples) * 100,
                'date_range': test_manifest['date_range'],
                'label_distribution': test_manifest['label_distribution']
            }
        }
        
        # Save summary
        summary_file = Path("data/metadata/split_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n" + "=" * 60)
        print("SPLIT SUMMARY")
        print("=" * 60)
        print(f"Total samples: {summary['total_samples']}")
        print(f"Train samples: {summary['train']['samples']} ({summary['train']['percentage']:.1f}%)")
        print(f"Test samples:  {summary['test']['samples']} ({summary['test']['percentage']:.1f}%)")
        
        print(f"\nTrain label distribution:")
        for label, count in summary['train']['label_distribution'].items():
            percentage = count / summary['train']['samples'] * 100
            print(f"  Label {label}: {count:3d} samples ({percentage:5.1f}%)")
        
        print(f"\nTest label distribution:")
        for label, count in summary['test']['label_distribution'].items():
            percentage = count / summary['test']['samples'] * 100
            print(f"  Label {label}: {count:3d} samples ({percentage:5.1f}%)")
        
        print(f"\nFiles saved to:")
        print(f"  Training: data/train/")
        print(f"  Testing:  data/test/")
        print(f"  Summary:  {summary_file}")
        
        return summary

def main():
    """Main execution function."""
    splitter = DatasetSplitter()
    summary = splitter.create_split()
    
    print(f"\n✅ Dataset split completed successfully!")
    print(f"Ready for ViT training with {summary['train']['samples']} training samples")

if __name__ == "__main__":
    main()