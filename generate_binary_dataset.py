#!/usr/bin/env python3
"""
Generate Binary Dataset - Main Script
Creates complete multi-instrument binary classification dataset for NQ_AI
Now supports Excel data source with DOW, NASDAQ, SP500 selection
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from binary_dataset_creator import BinaryDatasetCreator


def main():
    """Main dataset generation function."""
    parser = argparse.ArgumentParser(
        description="Generate Binary Classification Dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_binary_visual.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated list of instruments (DOW, NASDAQ, SP500, ALL). Example: 'DOW,NASDAQ' or 'ALL'",
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=None,
        help="Override start year for data collection",
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=None,
        help="Override end year for data collection",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry run without generating actual data",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for dataset generation (max 32, default: single-threaded)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing with auto-detected worker count",
    )

    args = parser.parse_args()

    try:
        print("🚀 Binary Dataset Generation for NQ_AI")
        print("=" * 50)

        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        print(f"📋 Configuration: {config['system']['name']}")
        print(f"📊 Classification: {config['classification']['classification_type']}")
        print(f"🎯 Classes: {list(config['classification']['labels'].values())}")

        # Override config if specified
        if args.start_year:
            config["data"]["start_year"] = args.start_year
        if args.end_year:
            config["data"]["end_year"] = args.end_year

        # Determine instruments
        if args.instruments:
            # Parse comma-separated instrument list
            instruments = [inst.strip().upper() for inst in args.instruments.split(",")]
            
            # Validate instruments
            available = config["data"]["available_instruments"]
            invalid_instruments = [inst for inst in instruments if inst not in available]
            if invalid_instruments:
                raise ValueError(f"Invalid instruments: {invalid_instruments}. Available: {available}")
            
            if len(instruments) == 1:
                print(f"🎲 Single instrument mode: {instruments[0]}")
            else:
                print(f"🌍 Multi-instrument mode: {instruments}")
        else:
            instruments = config["data"]["instruments"]
            print(f"🌍 Default multi-instrument mode: {instruments}")
        
        # Handle 'ALL' selection for display
        if "ALL" in instruments:
            available = [inst for inst in config["data"]["available_instruments"] if inst != "ALL"]
            print(f"🌍 ALL instruments selected: {available}")

        print(
            f"📅 Date range: {config['data']['start_year']}-{config['data']['end_year']}"
        )
        print(f"📈 Expected samples: ~{config['dataset']['expected_total_samples']:,}")

        if args.dry_run:
            print("\n🧪 DRY RUN MODE - No data will be generated")
            print("✅ Configuration validated successfully")
            print("🚀 Ready for full dataset generation")
            return 0

        # Determine worker count for parallel processing
        workers = None
        if args.parallel or args.workers:
            import multiprocessing as mp
            if args.workers:
                workers = min(args.workers, 32)  # Cap at 32 as requested
            else:
                workers = min(mp.cpu_count(), 32)  # Auto-detect, capped at 32
            
            if workers > 1:
                print(f"⚡ Parallel processing enabled: {workers} workers (max 32)")
            else:
                print(f"🔄 Single-threaded processing (workers={workers})")
        else:
            print(f"🔄 Single-threaded processing (default)")

        # Initialize dataset creator
        print(f"\n🏗️  Initializing Binary Dataset Creator...")
        creator = BinaryDatasetCreator(args.config, workers=workers)

        # Generate dataset
        if len(instruments) == 1 and "ALL" not in instruments:
            print(f"\n📊 Generating dataset for {instruments[0]}...")
            dataset_info = creator.create_instrument_dataset(
                instruments[0], 
                start_year=args.start_year, 
                end_year=args.end_year
            )

            print(f"✅ Dataset generated successfully!")
            print(f"   Samples: {dataset_info['total_samples']}")
            print(f"   Pattern distribution: {dataset_info['pattern_percentages']}")

        else:
            print(f"\n📊 Generating multi-instrument dataset...")
            dataset_manifest = creator.create_multi_instrument_dataset(
                instruments, 
                start_year=args.start_year, 
                end_year=args.end_year
            )

            print(f"✅ Multi-instrument dataset generated successfully!")
            print(f"   Total samples: {dataset_manifest['total_samples']:,}")
            print(f"   Instruments: {list(dataset_manifest['instruments'].keys())}")

            # Show instrument breakdown
            print(f"\n📈 Instrument Breakdown:")
            for instrument, info in dataset_manifest["instruments"].items():
                print(f"   {instrument}: {info['total_samples']:,} samples")
                print(f"     Pattern distribution: {info['pattern_distribution']}")

            # Show overall statistics
            stats = dataset_manifest["statistics"]
            print(f"\n📊 Overall Statistics:")
            print(f"   Bearish: {stats['pattern_percentages'][0]:.1f}%")
            print(f"   Bullish: {stats['pattern_percentages'][1]:.1f}%")
            print(f"   Neutral: {stats['pattern_percentages'][2]:.1f}%")

            # Show data splits
            splits = dataset_manifest["data_splits"]
            print(f"\n🔄 Data Splits:")
            print(
                f"   Train: {splits['train']['samples']:,} samples ({splits['train']['date_range']['start'][:10]} to {splits['train']['date_range']['end'][:10]})"
            )
            print(
                f"   Validation: {splits['validation']['samples']:,} samples ({splits['validation']['date_range']['start'][:10]} to {splits['validation']['date_range']['end'][:10]})"
            )
            print(
                f"   Test: {splits['test']['samples']:,} samples ({splits['test']['date_range']['start'][:10]} to {splits['test']['date_range']['end'][:10]})"
            )

        print(f"\n💾 Dataset Files:")
        print(f"   Images: {config['paths']['images']}")
        print(f"   Labels: {config['paths']['labels']}")
        print(f"   Metadata: {config['paths']['metadata']}")

        print(f"\n🎯 Next Steps:")
        print(f"   1. Verify dataset integrity")
        print(f"   2. Train binary ViT model:")
        print(f"      python src/train_binary_vit.py")
        print(f"   3. Deploy to RunPod for GPU training")

        print(f"\n✅ Binary dataset generation completed successfully!")

        return 0

    except KeyboardInterrupt:
        print(f"\n⏹️  Dataset generation interrupted by user")
        if 'creator' in locals():
            print("🧹 Cleaning up worker processes...")
            # Any necessary cleanup will be handled by ProcessPoolExecutor context manager
        return 1

    except Exception as e:
        print(f"\n❌ Dataset generation failed: {e}")
        import traceback

        traceback.print_exc()
        if 'creator' in locals():
            print("🧹 Cleaning up after error...")
        return 1


def verify_dataset():
    """Verify dataset integrity and statistics."""
    print("🔍 Verifying Binary Dataset Integrity...")

    try:
        # Load dataset manifest
        manifest_file = "data/metadata_binary/binary_dataset_manifest.json"
        if not os.path.exists(manifest_file):
            print("❌ Dataset manifest not found. Generate dataset first.")
            return 1

        import json

        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        total_samples = manifest["total_samples"]
        print(f"📊 Total samples: {total_samples:,}")

        # Verify image files exist
        missing_images = 0
        sample_check_count = min(100, total_samples)  # Check first 100 samples

        for i, sample in enumerate(manifest["samples"][:sample_check_count]):
            if not os.path.exists(sample["chart_path"]):
                missing_images += 1

        if missing_images > 0:
            print(
                f"⚠️  Warning: {missing_images}/{sample_check_count} checked images are missing"
            )
        else:
            print(f"✅ Image files verified ({sample_check_count} samples checked)")

        # Show statistics
        stats = manifest["statistics"]
        print(f"📈 Pattern Distribution:")
        print(f"   Bearish: {stats['pattern_percentages'][0]:.1f}%")
        print(f"   Bullish: {stats['pattern_percentages'][1]:.1f}%")
        print(f"   Neutral: {stats['pattern_percentages'][2]:.1f}%")

        print(f"✅ Dataset verification completed")
        return 0

    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        return 1


def cleanup_dataset():
    """Clean up old dataset files."""
    print("🧹 Cleaning up old dataset files...")

    try:
        import shutil

        # Clean up directories
        cleanup_dirs = [
            "data/images_binary",
            "data/labels_binary",
            "data/metadata_binary",
        ]

        removed_files = 0
        for directory in cleanup_dirs:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                removed_files += 1
                print(f"   Removed: {directory}")

        if removed_files == 0:
            print("   No old dataset files found")
        else:
            print(f"✅ Cleanup completed: {removed_files} directories removed")

        return 0

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return 1


if __name__ == "__main__":
    # Check for special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "verify":
            exit(verify_dataset())
        elif sys.argv[1] == "cleanup":
            exit(cleanup_dataset())

    # Run main dataset generation
    exit(main())
