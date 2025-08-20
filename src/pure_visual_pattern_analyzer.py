#!/usr/bin/env python3
"""
Pure Visual Pattern Analyzer for 4-Class Previous Day Levels Classification
Analyzes daily trading patterns WITHOUT numerical feature extraction
Pure visual approach - only pattern classification for dataset creation
"""

import pandas as pd
import numpy as np
import logging
import os
import yaml
from datetime import datetime
from typing import Dict, Tuple, Optional


class PureVisualPatternAnalyzer:
    """Pure visual pattern analyzer for 4-class daily pattern analysis - NO numerical features."""

    def __init__(self, config_path: str = "config/config_pure_visual_daily.yaml"):
        """Initialize pure visual pattern analyzer."""
        self.config = self.load_config(config_path)
        self.classification_config = self.config["classification"]
        self.setup_logging()

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging for pattern analysis."""
        log_dir = self.config["paths"]["logs_dir"]
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(f"{log_dir}/pure_visual_pattern_analyzer.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def analyze_daily_pattern(
        self, current_candle: pd.Series, previous_candle: pd.Series
    ) -> int:
        """
        Analyze 4-class daily pattern based on current vs previous day levels.
        This is the SAME logic as hybrid but WITHOUT numerical feature extraction.

        Args:
            current_candle: Current day OHLC data
            previous_candle: Previous day OHLC data

        Returns:
            Pattern classification (1-4):
            1: High Breakout - current_high >= prev_high && current_low > prev_low
            2: Low Breakdown - current_low <= prev_low && current_high < prev_high  
            3: Range Expansion - current_high >= prev_high && current_low <= prev_low
            4: Range Bound - current_high < prev_high && current_low > prev_low
        """
        try:
            # Extract current day levels
            current_high = float(current_candle["High"])
            current_low = float(current_candle["Low"])

            # Extract previous day levels
            prev_high = float(previous_candle["High"])
            prev_low = float(previous_candle["Low"])

            # Apply 4-class pattern analysis logic
            high_breakout = current_high >= prev_high
            low_breakdown = current_low <= prev_low
            high_respect = current_high < prev_high
            low_respect = current_low > prev_low

            # Determine pattern based on level interactions
            if high_breakout and low_respect:
                # High breakout: broke above previous high but respected previous low
                pattern = 1  # High Breakout
                pattern_name = "High Breakout"
            elif low_breakdown and high_respect:
                # Low breakdown: broke below previous low but respected previous high
                pattern = 2  # Low Breakdown
                pattern_name = "Low Breakdown"
            elif high_breakout and low_breakdown:
                # Range expansion: broke both previous high and low
                pattern = 3  # Range Expansion
                pattern_name = "Range Expansion"
            elif high_respect and low_respect:
                # Range bound: respected both previous high and low
                pattern = 4  # Range Bound
                pattern_name = "Range Bound"
            else:
                # This should not happen with proper logic, but handle edge case
                self.logger.warning(
                    f"Unexpected pattern combination: high_breakout={high_breakout}, "
                    f"low_breakdown={low_breakdown}, high_respect={high_respect}, low_respect={low_respect}"
                )
                pattern = 4  # Default to Range Bound

            self.logger.debug(
                f"Pattern analysis: {pattern_name} (Pattern {pattern})"
            )
            self.logger.debug(
                f"Current: H={current_high:.2f}, L={current_low:.2f}"
            )
            self.logger.debug(
                f"Previous: H={prev_high:.2f}, L={prev_low:.2f}"
            )

            return pattern

        except Exception as e:
            self.logger.error(f"Error analyzing pattern: {e}")
            raise

    def extract_metadata_features(
        self, current_candle: pd.Series, previous_candle: pd.Series
    ) -> Dict:
        """
        Extract ONLY metadata features for dataset manifest - NO numerical features.
        This is for pure visual approach - we only keep basic OHLC info for records.

        Args:
            current_candle: Current day OHLC data
            previous_candle: Previous day OHLC data

        Returns:
            Dictionary with metadata only (NO numerical features)
        """
        try:
            # Extract current day OHLC
            current_open = float(current_candle["Open"])
            current_high = float(current_candle["High"])
            current_low = float(current_candle["Low"])
            current_close = float(current_candle["Close"])
            current_volume = float(current_candle.get("Volume", 0))

            # Extract previous day OHLC
            prev_open = float(previous_candle["Open"])
            prev_high = float(previous_candle["High"])
            prev_low = float(previous_candle["Low"])
            prev_close = float(previous_candle["Close"])
            prev_volume = float(previous_candle.get("Volume", 0))

            # Calculate basic price movements (for metadata only)
            daily_range = current_high - current_low
            prev_daily_range = prev_high - prev_low
            price_change = current_close - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0

            # Create metadata features (NO numerical features for model input)
            metadata_features = {
                # Current day OHLC (for records)
                "current_ohlc": {
                    "open": current_open,
                    "high": current_high,
                    "low": current_low,
                    "close": current_close,
                    "volume": current_volume,
                    "range": daily_range,
                },
                # Previous day OHLC (for records)
                "previous_ohlc": {
                    "open": prev_open,
                    "high": prev_high,
                    "low": prev_low,
                    "close": prev_close,
                    "volume": prev_volume,
                    "range": prev_daily_range,
                },
                # Price movement metadata (for records)
                "price_movement": {
                    "change": price_change,
                    "change_pct": price_change_pct,
                    "daily_range": daily_range,
                    "prev_range": prev_daily_range,
                },
                # Level interaction flags (for records)
                "level_interactions": {
                    "high_breakout": current_high >= prev_high,
                    "low_breakdown": current_low <= prev_low,
                    "high_respect": current_high < prev_high,
                    "low_respect": current_low > prev_low,
                },
                # Pure visual specific
                "analysis_type": "pure_visual",
                "numerical_features_extracted": False,
                "visual_only": True,
            }

            return metadata_features

        except Exception as e:
            self.logger.error(f"Error extracting metadata features: {e}")
            raise

    def get_pattern_statistics(self, patterns: list) -> Dict:
        """
        Calculate pattern distribution statistics for 4-class system.

        Args:
            patterns: List of pattern classifications (1-4)

        Returns:
            Dictionary with pattern statistics
        """
        try:
            if not patterns:
                return {}

            # Count patterns
            pattern_counts = {}
            for pattern in range(1, 5):  # Patterns 1-4
                pattern_counts[pattern] = patterns.count(pattern)

            total_patterns = len(patterns)
            pattern_percentages = {}
            for pattern, count in pattern_counts.items():
                pattern_percentages[pattern] = (count / total_patterns) * 100

            # Pattern labels
            pattern_labels = {
                1: "High Breakout",
                2: "Low Breakdown",
                3: "Range Expansion", 
                4: "Range Bound"
            }

            # Calculate balance metrics
            most_common = max(pattern_counts, key=pattern_counts.get)
            least_common = min(pattern_counts, key=pattern_counts.get)
            balance_ratio = (
                pattern_counts[least_common] / pattern_counts[most_common]
                if pattern_counts[most_common] > 0 else 0
            )

            statistics = {
                "total_patterns": total_patterns,
                "pattern_counts": pattern_counts,
                "pattern_percentages": pattern_percentages,
                "pattern_labels": pattern_labels,
                "balance_metrics": {
                    "most_common_pattern": most_common,
                    "least_common_pattern": least_common,
                    "balance_ratio": balance_ratio,
                    "is_balanced": balance_ratio >= 0.8,  # Consider balanced if ratio >= 0.8
                },
                "analysis_type": "pure_visual_4class",
            }

            # Log statistics
            self.logger.info(f"Pattern Statistics (Pure Visual):")
            self.logger.info(f"  Total patterns: {total_patterns}")
            for pattern, count in pattern_counts.items():
                label = pattern_labels[pattern]
                percentage = pattern_percentages[pattern]
                self.logger.info(f"  Pattern {pattern} ({label}): {count} ({percentage:.1f}%)")
            self.logger.info(f"  Balance ratio: {balance_ratio:.3f}")

            return statistics

        except Exception as e:
            self.logger.error(f"Error calculating pattern statistics: {e}")
            raise

    def validate_pattern_data(
        self, current_candle: pd.Series, previous_candle: pd.Series
    ) -> bool:
        """
        Validate that pattern analysis data is complete and valid.

        Args:
            current_candle: Current day OHLC data
            previous_candle: Previous day OHLC data

        Returns:
            True if data is valid for pattern analysis
        """
        try:
            # Check required columns
            required_columns = ["Open", "High", "Low", "Close"]
            
            for col in required_columns:
                if col not in current_candle.index or pd.isna(current_candle[col]):
                    self.logger.error(f"Current candle missing or invalid {col}")
                    return False
                    
                if col not in previous_candle.index or pd.isna(previous_candle[col]):
                    self.logger.error(f"Previous candle missing or invalid {col}")
                    return False

            # Check OHLC relationships for current candle
            current_high = float(current_candle["High"])
            current_low = float(current_candle["Low"])
            current_open = float(current_candle["Open"])
            current_close = float(current_candle["Close"])

            if not (
                current_low <= current_open <= current_high
                and current_low <= current_close <= current_high
            ):
                self.logger.error("Invalid OHLC relationship in current candle")
                return False

            # Check OHLC relationships for previous candle
            prev_high = float(previous_candle["High"])
            prev_low = float(previous_candle["Low"])
            prev_open = float(previous_candle["Open"])
            prev_close = float(previous_candle["Close"])

            if not (
                prev_low <= prev_open <= prev_high
                and prev_low <= prev_close <= prev_high
            ):
                self.logger.error("Invalid OHLC relationship in previous candle")
                return False

            # Check for reasonable price values (not zero or negative)
            all_prices = [
                current_high, current_low, current_open, current_close,
                prev_high, prev_low, prev_open, prev_close
            ]
            
            if any(price <= 0 for price in all_prices):
                self.logger.error("Invalid price values (zero or negative)")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating pattern data: {e}")
            return False

    def create_pattern_summary(
        self,
        pattern: int,
        current_candle: pd.Series,
        previous_candle: pd.Series,
        analysis_date: datetime = None,
    ) -> Dict:
        """
        Create comprehensive pattern summary for pure visual analysis.

        Args:
            pattern: Pattern classification (1-4)
            current_candle: Current day OHLC data
            previous_candle: Previous day OHLC data
            analysis_date: Date of analysis

        Returns:
            Dictionary with pattern summary
        """
        try:
            # Pattern labels
            pattern_labels = {
                1: "High Breakout",
                2: "Low Breakdown",
                3: "Range Expansion",
                4: "Range Bound"
            }

            # Extract metadata features
            metadata = self.extract_metadata_features(current_candle, previous_candle)

            # Create comprehensive summary
            summary = {
                "analysis_date": analysis_date.isoformat() if analysis_date else None,
                "pattern": pattern,
                "pattern_label": pattern_labels.get(pattern, f"Unknown Pattern {pattern}"),
                "classification_type": "pure_visual_4class_previous_day_levels",
                "metadata": metadata,
                "pure_visual_config": {
                    "numerical_features": False,
                    "visual_only": True,
                    "image_resolution": f"{self.config['chart']['image_size']}x{self.config['chart']['image_size']}",
                    "volume_bars": self.config['chart']['volume'],
                    "reference_lines": self.config['chart']['show_prev_day_lines'],
                },
                "validation": {
                    "data_valid": self.validate_pattern_data(current_candle, previous_candle),
                    "pattern_logic_applied": True,
                    "analysis_type": "pure_visual",
                }
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error creating pattern summary: {e}")
            raise


def main():
    """Test the pure visual pattern analyzer."""
    print("🚀 NQ_AI Pure Visual Pattern Analyzer")
    print("4-Class Previous Day Levels Classification - Pure Visual Analysis")

    try:
        # Initialize analyzer
        analyzer = PureVisualPatternAnalyzer()

        print("🏗️  Initializing pure visual pattern analyzer...")
        print(f"   📊 Classification: 4-class previous day levels")
        print(f"   🎯 Approach: Pure visual (NO numerical features)")

        # Create test data
        print("\n🧪 Testing with sample data...")

        # Test case 1: High Breakout pattern
        current_1 = pd.Series({
            "Open": 15000,
            "High": 15150,  # Above prev high (15100)
            "Low": 14950,   # Above prev low (14900)
            "Close": 15120,
            "Volume": 1000000
        })
        previous_1 = pd.Series({
            "Open": 14950,
            "High": 15100,
            "Low": 14900,
            "Close": 15050,
            "Volume": 800000
        })

        pattern_1 = analyzer.analyze_daily_pattern(current_1, previous_1)
        print(f"   Test 1 - High Breakout: Pattern {pattern_1} (Expected: 1)")

        # Test case 2: Low Breakdown pattern
        current_2 = pd.Series({
            "Open": 15000,
            "High": 15050,  # Below prev high (15100)
            "Low": 14850,   # Below prev low (14900)
            "Close": 14920,
            "Volume": 1200000
        })
        previous_2 = pd.Series({
            "Open": 14950,
            "High": 15100,
            "Low": 14900,
            "Close": 15050,
            "Volume": 800000
        })

        pattern_2 = analyzer.analyze_daily_pattern(current_2, previous_2)
        print(f"   Test 2 - Low Breakdown: Pattern {pattern_2} (Expected: 2)")

        # Test case 3: Range Expansion pattern
        current_3 = pd.Series({
            "Open": 15000,
            "High": 15150,  # Above prev high (15100)
            "Low": 14850,   # Below prev low (14900)
            "Close": 15020,
            "Volume": 1500000
        })
        previous_3 = pd.Series({
            "Open": 14950,
            "High": 15100,
            "Low": 14900,
            "Close": 15050,
            "Volume": 800000
        })

        pattern_3 = analyzer.analyze_daily_pattern(current_3, previous_3)
        print(f"   Test 3 - Range Expansion: Pattern {pattern_3} (Expected: 3)")

        # Test case 4: Range Bound pattern
        current_4 = pd.Series({
            "Open": 15000,
            "High": 15050,  # Below prev high (15100)
            "Low": 14950,   # Above prev low (14900)
            "Close": 15020,
            "Volume": 900000
        })
        previous_4 = pd.Series({
            "Open": 14950,
            "High": 15100,
            "Low": 14900,
            "Close": 15050,
            "Volume": 800000
        })

        pattern_4 = analyzer.analyze_daily_pattern(current_4, previous_4)
        print(f"   Test 4 - Range Bound: Pattern {pattern_4} (Expected: 4)")

        # Test metadata extraction (NO numerical features)
        print(f"\n📊 Testing metadata extraction...")
        metadata = analyzer.extract_metadata_features(current_1, previous_1)
        print(f"   Metadata extracted: {len(metadata)} categories")
        print(f"   Numerical features extracted: {metadata['numerical_features_extracted']}")
        print(f"   Visual only: {metadata['visual_only']}")
        print(f"   Analysis type: {metadata['analysis_type']}")

        # Test pattern statistics
        print(f"\n📈 Testing pattern statistics...")
        test_patterns = [1, 1, 2, 3, 4, 1, 2, 3, 4, 4]  # Sample pattern distribution
        stats = analyzer.get_pattern_statistics(test_patterns)
        print(f"   Total patterns: {stats['total_patterns']}")
        print(f"   Pattern distribution:")
        for pattern, count in stats['pattern_counts'].items():
            label = stats['pattern_labels'][pattern]
            percentage = stats['pattern_percentages'][pattern]
            print(f"     Pattern {pattern} ({label}): {count} ({percentage:.1f}%)")

        # Test pattern summary creation
        print(f"\n📋 Testing pattern summary creation...")
        summary = analyzer.create_pattern_summary(
            pattern_1, current_1, previous_1, datetime.now()
        )
        print(f"   Pattern: {summary['pattern']} ({summary['pattern_label']})")
        print(f"   Classification type: {summary['classification_type']}")
        print(f"   Pure visual config: {summary['pure_visual_config']}")

        print("\n🎯 Pure Visual Pattern Analyzer Features:")
        print("   - 4-class previous day levels classification")
        print("   - NO numerical feature extraction (pure visual)")
        print("   - Metadata extraction for dataset records only")
        print("   - Pattern validation and statistics")
        print("   - Same pattern logic as hybrid but visual-only")
        print("   - Compatible with 448x448 chart generation")

        print("✅ Pure visual pattern analyzer test completed successfully!")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())