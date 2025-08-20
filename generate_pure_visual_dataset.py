#!/usr/bin/env python3
"""
Generate Pure Visual Dataset for "The Test"
Creates 448x448 chart images with NO numerical features
Tests if ViT can learn patterns without numerical crutch
"""

import sys
import os
sys.path.append("src")

import argparse
import logging
from datetime import datetime
from typing import List, Optional

# Import the dataset components
from excel_data_fetcher import ExcelDataFetcher
from daily_chart_generator import DailyChartGenerator  
from daily_pattern_analyzer import DailyPatternAnalyzer
from daily_dataset_creator import DailyDatasetCreator


def parse_arguments():
    """Parse command line arguments for pure visual dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate Pure Visual Dataset for 'The Test' - 448x448 charts, NO numerical features"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_pure_visual_daily.yaml",
        help="Configuration file path for pure visual learning",
    )
    
    parser.add_argument(
        "--instruments",
        type=str,
        default="NASDAQ",
        help="Instrument to process (NASDAQ for 'The Test')",
    )
    
    parser.add_argument(
        "--start_year",
        type=int,
        default=2000,
        help="Override start year (default: 2000)",
    )
    
    parser.add_argument(
        "--end_year", 
        type=int,
        default=2025,
        help="Override end year (default: 2025)",
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for faster generation",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8 for pure visual)",
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true", 
        help="Validate configuration without generating dataset",
    )
    
    return parser.parse_args()


def setup_logging():
    """Setup logging for pure visual dataset generation."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/pure_visual_dataset_generation.log"),
            logging.StreamHandler(),
        ],
    )
    
    return logging.getLogger(__name__)


def validate_pure_visual_config(config_path: str, logger) -> bool:
    """Validate pure visual configuration for 'The Test'."""
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check pure visual requirements
        if config.get("features", {}).get("numerical_features", 3) != 0:
            logger.error("❌ Config error: numerical_features must be 0 for pure visual")
            return False
            
        if config.get("model", {}).get("visual_only") != True:
            logger.error("❌ Config error: visual_only must be True for pure visual")
            return False
            
        if config.get("model", {}).get("use_feature_fusion") != False:
            logger.error("❌ Config error: use_feature_fusion must be False for pure visual")
            return False
            
        image_size = config.get("chart", {}).get("image_size", 224)
        if image_size != 448:
            logger.warning(f"⚠️  Image size is {image_size}, recommend 448 for 'The Test'")
            
        # Check data source
        if config.get("data", {}).get("source") != "excel":
            logger.error("❌ Config error: data source must be 'excel' for RunPod")
            return False
            
        logger.info("✅ Pure visual configuration validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config validation error: {e}")
        return False


def main():
    """Main function for pure visual dataset generation."""
    print("🚀 NQ_AI Pure Visual Dataset Generator - 'The Test'")
    print("Testing visual learning without numerical crutch")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Starting PURE VISUAL dataset generation for 'The Test'")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Instruments: {args.instruments}")
    logger.info(f"Date range: {args.start_year}-{args.end_year}")
    logger.info(f"Parallel processing: {args.parallel} (workers: {args.workers})")
    
    try:
        # Validate pure visual configuration
        if not validate_pure_visual_config(args.config, logger):
            logger.error("❌ Configuration validation failed")
            return 1
            
        # Dry run check
        if args.dry_run:
            logger.info("🧪 DRY RUN: Configuration validated successfully")
            print("✅ Pure visual configuration is valid for 'The Test'")
            return 0
        
        # Initialize dataset creator with pure visual config
        logger.info("🏗️  Initializing pure visual dataset creator...")
        dataset_creator = DailyDatasetCreator(config_path=args.config)
        
        # Parse instruments (single instrument for 'The Test')
        instruments = [args.instruments] if args.instruments != "ALL" else ["NASDAQ"]
        
        # Override date range if specified
        override_config = {}
        if args.start_year != 2000:
            override_config["start_year"] = args.start_year
        if args.end_year != 2025:
            override_config["end_year"] = args.end_year
        
        # Generate pure visual dataset
        logger.info(f"📊 Generating PURE VISUAL dataset for {instruments}")
        logger.info("🔬 'The Test' Parameters:")
        logger.info("   - NO numerical features (pure visual)")
        logger.info("   - 448x448 high-resolution charts")
        logger.info("   - Previous day level reference lines")
        logger.info("   - 4-class pattern classification")
        logger.info("   - Success: >40% accuracy proves visual learning")
        
        # Create dataset with parallel processing if enabled
        if args.parallel:
            logger.info(f"⚡ Using parallel processing with {args.workers} workers")
            results = dataset_creator.create_multi_instrument_dataset_parallel(
                instruments=instruments,
                override_config=override_config,
                max_workers=args.workers
            )
        else:
            logger.info("🔄 Using sequential processing")
            results = dataset_creator.create_multi_instrument_dataset(
                instruments=instruments,
                override_config=override_config
            )
        
        # Report results
        if results:
            total_samples = sum(len(samples) for samples in results.values())
            logger.info("🎯 PURE VISUAL dataset generation completed!")
            logger.info(f"   Total samples: {total_samples:,}")
            logger.info(f"   Instruments: {list(results.keys())}")
            logger.info(f"   Charts: 448x448 pixels with reference lines")
            logger.info(f"   Numerical features: 0 (PURE VISUAL)")
            
            print(f"\n✅ Pure Visual Dataset Generation Complete!")
            print(f"📊 Total samples: {total_samples:,}")
            print(f"🎯 Ready for 'The Test' - pure visual ViT training")
            print(f"🔬 Success criteria: >40% accuracy proves visual learning")
            
        else:
            logger.error("❌ Dataset generation failed")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Dataset generation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Dataset generation error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("🏁 Pure visual dataset generation process completed")
    return 0


if __name__ == "__main__":
    exit(main())