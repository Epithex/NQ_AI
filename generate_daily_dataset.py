#!/usr/bin/env python3
"""
Generate 4-Class Hybrid Daily Dataset - Main Script
Creates complete multi-instrument hybrid classification dataset for NQ_AI
Supports Excel data source with DOW, NASDAQ, SP500 selection and parallel processing
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

from daily_dataset_creator import DailyDatasetCreator


def main():
    """Main dataset generation function."""
    parser = argparse.ArgumentParser(
        description="Generate 4-Class Hybrid Daily Classification Dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_daily_hybrid.yaml",
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
        print("🚀 4-Class Hybrid Daily Dataset Generation for NQ_AI")
        print("=" * 55)

        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        print(f"📋 Configuration: {config['system']['name']}")
        print(f"📊 Classification: {config['classification']['classification_type']}")
        print(f"🎯 Classes: {list(config['classification']['labels'].values())}")
        print(f"🔧 Model Type: {config['system']['architecture']}")

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
            print("🚀 Ready for full hybrid dataset generation")
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
        print(f"\n🏗️  Initializing 4-Class Hybrid Dataset Creator...")
        creator = DailyDatasetCreator(args.config, workers=workers)

        # Generate dataset
        print(f"\n📊 Generating 4-class hybrid dataset...")
        result = creator.create_multi_instrument_dataset(
            instruments, 
            start_year=args.start_year, 
            end_year=args.end_year
        )

        if result["success"]:
            print(f"✅ 4-class hybrid dataset generated successfully!")
            print(f"   Total samples: {result['total_samples']:,}")
            print(f"   Instruments: {result['instruments']}")
            print(f"   Model type: {result['model_type']}")
            print(f"   Visual features: {result['features']['visual']}")
            print(f"   Numerical features: {result['features']['numerical']}")

            # Show statistics if available
            if hasattr(creator, 'dataset_manifest') and creator.dataset_manifest.get("statistics"):
                stats = creator.dataset_manifest["statistics"]
                print(f"\n📊 Pattern Distribution:")
                for pattern_id, count in stats["pattern_counts"].items():
                    label = stats["pattern_labels"][pattern_id]
                    percentage = stats["pattern_percentages"][pattern_id]
                    print(f"   {pattern_id}: {label} - {count:,} ({percentage:.1f}%)")
                
                # Show class balance information
                balance = stats["class_balance"]
                print(f"\n⚖️  Class Balance:")
                print(f"   Most common: Pattern {balance['most_common']}")
                print(f"   Least common: Pattern {balance['least_common']}")
                print(f"   Balance ratio: {balance['balance_ratio']:.3f}")

        else:
            print(f"❌ Dataset generation failed: {result['error']}")
            return 1

        print(f"\n💾 Dataset Files:")
        print(f"   Images: {config['paths']['images']}")
        print(f"   Labels: {config['paths']['labels']}")
        print(f"   Metadata: {config['paths']['metadata']}")

        print(f"\n🎯 Next Steps:")
        print(f"   1. Verify dataset integrity:")
        print(f"      python generate_daily_dataset.py verify")
        print(f"   2. Train 4-class hybrid ViT model:")
        print(f"      python src/train_daily_model.py")
        print(f"   3. Deploy to RunPod for GPU training")

        print(f"\n✅ 4-class hybrid dataset generation completed successfully!")

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
    """Verify hybrid dataset integrity and statistics."""
    print("🔍 Verifying 4-Class Hybrid Dataset Integrity...")

    try:
        # Look for the latest dataset manifest
        import glob
        import json
        
        manifest_pattern = "data/metadata_daily/daily_hybrid_dataset_manifest_*.json"
        manifest_files = glob.glob(manifest_pattern)
        
        if not manifest_files:
            print("❌ Dataset manifest not found. Generate dataset first.")
            print(f"   Looking for: {manifest_pattern}")
            return 1
        
        # Get the most recent manifest
        latest_manifest = max(manifest_files, key=os.path.getctime)
        print(f"📄 Using manifest: {latest_manifest}")

        with open(latest_manifest, "r") as f:
            manifest = json.load(f)

        total_samples = manifest["total_samples"]
        print(f"📊 Total samples: {total_samples:,}")
        print(f"🔧 Model type: {manifest['model_type']}")

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

        # Show 4-class statistics
        stats = manifest["statistics"]
        print(f"📈 4-Class Pattern Distribution:")
        for pattern_id, count in stats["pattern_counts"].items():
            label = stats["pattern_labels"][pattern_id]
            percentage = stats["pattern_percentages"][pattern_id]
            print(f"   {pattern_id}: {label} - {count:,} ({percentage:.1f}%)")

        # Show class balance
        balance = stats["class_balance"]
        print(f"\n⚖️  Class Balance Analysis:")
        print(f"   Most common: Pattern {balance['most_common']} ({stats['pattern_labels'][balance['most_common']]})")
        print(f"   Least common: Pattern {balance['least_common']} ({stats['pattern_labels'][balance['least_common']]})")
        print(f"   Balance ratio: {balance['balance_ratio']:.3f}")

        # Show feature information
        if "features" in manifest:
            features = manifest["features"]
            print(f"\n🔧 Feature Configuration:")
            print(f"   Visual: {features['visual']}")
            print(f"   Numerical: {features['numerical']}")

        print(f"✅ 4-class hybrid dataset verification completed")
        return 0

    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cleanup_dataset():
    """Clean up old hybrid dataset files."""
    print("🧹 Cleaning up old 4-class hybrid dataset files...")

    try:
        import shutil

        # Clean up directories
        cleanup_dirs = [
            "data/images_daily",
            "data/labels_daily",
            "data/metadata_daily",
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


def show_help():
    """Show detailed help information."""
    print("🚀 4-Class Hybrid Daily Dataset Generator")
    print("=" * 50)
    print()
    print("USAGE:")
    print("  python generate_daily_dataset.py [OPTIONS]")
    print("  python generate_daily_dataset.py verify")
    print("  python generate_daily_dataset.py cleanup")
    print("  python generate_daily_dataset.py help")
    print()
    print("DATASET GENERATION OPTIONS:")
    print("  --instruments INSTRUMENTS    Comma-separated list (DOW, NASDAQ, SP500, ALL)")
    print("  --start_year YEAR           Override start year (default: 2000)")
    print("  --end_year YEAR             Override end year (default: 2025)")
    print("  --config CONFIG             Custom configuration file")
    print("  --dry_run                   Validate configuration without generating data")
    print("  --parallel                  Enable parallel processing (auto-detect workers)")
    print("  --workers N                 Specify number of workers (max 32)")
    print()
    print("SPECIAL COMMANDS:")
    print("  verify                      Verify dataset integrity")
    print("  cleanup                     Remove old dataset files")
    print("  help                        Show this help message")
    print()
    print("EXAMPLES:")
    print("  # Generate dataset for all instruments")
    print("  python generate_daily_dataset.py --instruments ALL")
    print()
    print("  # Generate with parallel processing")
    print("  python generate_daily_dataset.py --instruments ALL --parallel")
    print()
    print("  # Generate for specific instruments and date range")
    print("  python generate_daily_dataset.py --instruments DOW,NASDAQ --start_year 2020 --end_year 2024")
    print()
    print("  # Test configuration without generating data")
    print("  python generate_daily_dataset.py --instruments NASDAQ --dry_run")
    print()
    print("4-CLASS PATTERN SYSTEM:")
    print("  1. High Breakout:    current_high >= prev_high && current_low > prev_low")
    print("  2. Low Breakdown:    current_low <= prev_low && current_high < prev_high")
    print("  3. Range Expansion:  current_high >= prev_high && current_low <= prev_low")
    print("  4. Range Bound:      current_high < prev_high && current_low > prev_low")
    print()
    print("HYBRID MODEL FEATURES:")
    print("  • Visual: 30-bar daily charts with previous day level reference lines")
    print("  • Numerical: 3 key features (distance_to_prev_high, distance_to_prev_low, prev_day_range)")
    print("  • Architecture: ViT-Base (87M+ params) with feature fusion")
    print("  • Multi-instrument: DOW, NASDAQ, SP500 futures data")


if __name__ == "__main__":
    # Check for special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "verify":
            exit(verify_dataset())
        elif sys.argv[1] == "cleanup":
            exit(cleanup_dataset())
        elif sys.argv[1] == "help":
            show_help()
            exit(0)

    # Run main dataset generation
    exit(main())