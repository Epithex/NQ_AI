#!/usr/bin/env python3
"""
NQ_AI Data Generation Pipeline
Main execution script for generating training datasets
"""

import sys
import argparse
from datetime import datetime
import yaml
import pandas as pd
from src.dataset_creator import DatasetCreator
import logging


def setup_logging():
    """Configure root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/main.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main execution function."""
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="NQ_AI Data Generation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["training", "validation", "test"],
        default="training",
        help="Type of dataset to generate",
    )
    parser.add_argument(
        "--start-date", type=str, help="Override config start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, help="Override config end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without generating data",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Get date range from config or command line
        if args.dataset in config["date_ranges"]:
            date_config = config["date_ranges"][args.dataset]
            start_date = args.start_date or date_config["start"]
            end_date = args.end_date or date_config["end"]
        else:
            if not args.start_date or not args.end_date:
                raise ValueError("Start and end dates required for custom dataset")
            start_date = args.start_date
            end_date = args.end_date

        logger.info(f"=" * 60)
        logger.info(f"NQ_AI Data Generation Pipeline")
        logger.info(f"Dataset Type: {args.dataset}")
        logger.info(f"Date Range: {start_date} to {end_date}")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"=" * 60)

        if args.dry_run:
            logger.info("DRY RUN MODE - Validating configuration only")
            # Validate dates
            try:
                pd.to_datetime(start_date)
                pd.to_datetime(end_date)
                logger.info("✓ Date validation passed")
            except:
                logger.error("✗ Invalid date format")
                return 1

            # Test component initialization
            try:
                creator = DatasetCreator(args.config)
                logger.info("✓ All components initialized successfully")
            except Exception as e:
                logger.error(f"✗ Component initialization failed: {str(e)}")
                return 1

            logger.info("Dry run completed successfully")
            return 0

        # Create dataset creator instance
        creator = DatasetCreator(args.config)

        # Generate dataset
        creator.generate_dataset(
            start_date=start_date, end_date=end_date, dataset_type=args.dataset
        )

        logger.info("Dataset generation completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
