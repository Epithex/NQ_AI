#!/usr/bin/env python3
"""
Generate Pure Visual Daily Dataset - Main Script
Creates complete multi-instrument pure visual classification dataset for NQ_AI
Uses ONLY 448x448 chart images - NO numerical features
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

from pure_visual_dataset_creator import PureVisualDatasetCreator


def main():
    """Main dataset generation function."""
    parser = argparse.ArgumentParser(
        description="Generate Pure Visual Daily Classification Dataset (NO numerical features)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_pure_visual_daily.yaml",
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
        print("🚀 Pure Visual Daily Dataset Generation for NQ_AI")
        print("=" * 55)

        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        print(f"📋 Configuration: {config['system']['name']}")
        print(f"📊 Classification: {config['classification']['classification_type']}")
        print(f"🎯 Classes: {list(config['classification']['labels'].values())}")
        print(f"🔧 Model Type: {config['system']['architecture']}")
        print(f"📈 Image Resolution: {config['chart']['image_size']}x{config['chart']['image_size']}")
        print(f"📦 Volume Bars: {'ENABLED' if config['chart']['volume'] else 'DISABLED'}")
        print(f"🚫 Numerical Features: NONE - Pure visual only")

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
                print(f"🎯 Target Instrument: {instruments[0]}")
            else:
                print(f"🎯 Target Instruments: {', '.join(instruments)}")
        else:
            instruments = None
            print(f"🎯 Target Instruments: {config['data']['instruments']} (default)")

        # Determine worker count for parallel processing
        workers = None
        if args.parallel:
            import multiprocessing
            workers = min(multiprocessing.cpu_count(), 32)  # Cap at 32 workers
            print(f"🏭 Parallel Processing: AUTO ({workers} workers)")
        elif args.workers:
            workers = min(args.workers, 32)  # Cap at 32 workers
            print(f"🏭 Parallel Processing: {workers} workers")
        else:
            print(f"🏭 Parallel Processing: DISABLED (single-threaded)")

        # Show data parameters
        start_year = args.start_year or config["data"]["start_year"]
        end_year = args.end_year or config["data"]["end_year"]
        print(f"📅 Date Range: {start_year} - {end_year}")
        print(f"📊 Chart Window: {config['data']['bars_per_chart']} daily bars")

        # Dry run check
        if args.dry_run:
            print("\n🧪 DRY RUN MODE - No data will be generated")
            print("✅ Configuration validation completed!")
            print("💡 Remove --dry_run flag to generate actual dataset")
            return 0

        print(f"\n🏗️  Initializing Pure Visual Dataset Creator...")

        # Initialize creator
        creator = PureVisualDatasetCreator(
            config_path=args.config,
            workers=workers
        )

        print(f"📊 Expected Features:")
        print(f"   - Chart Images: 448x448 pixels with volume bars")
        print(f"   - Reference Lines: Previous day high (green) and low (red)")
        print(f"   - Numerical Features: NONE (pure visual approach)")
        print(f"   - Pattern Classes: 4 (High Breakout, Low Breakdown, Range Expansion, Range Bound)")

        # Generate dataset
        print(f"\n🚀 Starting dataset generation...")
        start_time = datetime.now()

        result = creator.create_multi_instrument_dataset(
            instruments=instruments,
            start_year=start_year,
            end_year=end_year,
        )

        end_time = datetime.now()
        duration = end_time - start_time

        # Report results
        print(f"\n📊 Dataset Generation Results:")
        print(f"=" * 40)

        if result["success"]:
            print(f"✅ Status: SUCCESS")
            print(f"📈 Total Samples: {result['total_samples']:,}")
            print(f"🎯 Instruments: {', '.join(result['instruments'])}")
            print(f"📅 Date Range: {result['date_range']}")
            print(f"⏱️  Duration: {duration}")
            print(f"📁 Manifest: {result['manifest_path']}")

            # Show processing rate
            if duration.total_seconds() > 0:
                rate = result['total_samples'] / duration.total_seconds()
                print(f"🚀 Processing Rate: {rate:.2f} samples/second")

            # Show pattern distribution if available
            if creator.dataset_manifest.get("statistics"):
                stats = creator.dataset_manifest["statistics"]
                print(f"\n🎯 Pattern Distribution:")
                for pattern_id, count in stats["pattern_counts"].items():
                    label = stats["pattern_labels"][pattern_id]
                    percentage = stats["pattern_percentages"][pattern_id]
                    print(f"   {pattern_id}: {label} - {count:,} ({percentage:.1f}%)")

                balance_ratio = stats["class_balance"]["balance_ratio"]
                print(f"⚖️  Class Balance Ratio: {balance_ratio:.3f}")

            # Show storage information
            print(f"\n💾 Storage Information:")
            print(f"   📁 Images: {config['paths']['images']}")
            print(f"   📋 Labels: {config['paths']['labels']}")
            print(f"   📊 Metadata: {config['paths']['metadata']}")

            print(f"\n🎯 Pure Visual Dataset Features:")
            print(f"   - Resolution: 448x448 pixels (high quality)")
            print(f"   - Volume Bars: ENABLED for richer visual information")
            print(f"   - Reference Lines: Previous day high/low levels")
            print(f"   - Numerical Features: NONE - pure visual learning")
            print(f"   - Pattern Classification: 4-class previous day levels")
            print(f"   - Chart Context: 30 daily bars (excludes analysis day)")

            print(f"\n🚀 Ready for Pure Visual Training:")
            print(f"   python src/train_pure_visual_daily.py --config {args.config}")

        else:
            print(f"❌ Status: FAILED")
            print(f"🚨 Error: {result['error']}")
            return 1

        print(f"\n✅ Pure Visual Dataset Generation Completed!")

    except KeyboardInterrupt:
        print(f"\n🛑 Dataset generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())