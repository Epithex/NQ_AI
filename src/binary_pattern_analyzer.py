#!/usr/bin/env python3
"""
Binary Pattern Analyzer for NQ Trading
Analyzes daily candles for binary bullish/bearish classification based on open vs close
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import yaml
import logging
import os


class BinaryPatternAnalyzer:
    """Analyzes daily candles for binary bullish/bearish classification."""

    def __init__(self, config_path: str = "config/config_binary_visual.yaml"):
        """Initialize binary pattern analyzer with configuration."""
        self.config = self.load_config(config_path)
        self.classification_config = self.config["classification"]
        self.setup_logging()

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging for the analyzer."""
        log_dir = self.config["paths"]["logs_dir"]
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(f"{log_dir}/binary_pattern_analyzer.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def analyze_binary_pattern(self, daily_candle: pd.Series) -> int:
        """
        Classify daily pattern based on open vs close comparison.

        3-class binary classification system:
        0: Bearish - Close < Open (red candle)
        1: Bullish - Close > Open (green candle)
        2: Neutral - Close = Open (doji candle, very rare)

        Args:
            daily_candle: Daily OHLC data

        Returns:
            Pattern classification (0-2)
        """
        try:
            # Extract open and close prices
            open_price = float(daily_candle["Open"])
            close_price = float(daily_candle["Close"])

            # Calculate price difference
            price_diff = close_price - open_price

            # Classify based on open vs close
            if price_diff > 0:
                pattern = 1  # Bullish (green candle)
                description = "Bullish"
            elif price_diff < 0:
                pattern = 0  # Bearish (red candle)
                description = "Bearish"
            else:
                pattern = 2  # Neutral (doji candle)
                description = "Neutral"

            # Calculate additional metrics
            price_change_pct = (price_diff / open_price) * 100 if open_price != 0 else 0
            candle_body_size = abs(price_diff)

            self.logger.debug(f"Pattern {pattern}: {description}")
            self.logger.debug(f"Open: {open_price:.2f}, Close: {close_price:.2f}")
            self.logger.debug(
                f"Price change: {price_diff:.2f} ({price_change_pct:.3f}%)"
            )
            self.logger.debug(f"Candle body size: {candle_body_size:.2f}")

            return pattern

        except Exception as e:
            self.logger.error(f"Error analyzing binary pattern: {e}")
            raise

    def extract_candle_features(self, daily_candle: pd.Series) -> Dict[str, float]:
        """
        Extract candle-based numerical features for analysis.

        Args:
            daily_candle: Daily OHLC data

        Returns:
            Dictionary of candle features
        """
        try:
            open_price = float(daily_candle["Open"])
            high_price = float(daily_candle["High"])
            low_price = float(daily_candle["Low"])
            close_price = float(daily_candle["Close"])

            # Calculate candle metrics
            price_diff = close_price - open_price
            candle_range = high_price - low_price
            body_size = abs(price_diff)

            # Upper and lower shadows
            if close_price > open_price:  # Bullish candle
                upper_shadow = high_price - close_price
                lower_shadow = open_price - low_price
            else:  # Bearish candle
                upper_shadow = high_price - open_price
                lower_shadow = close_price - low_price

            features = {
                "price_change": price_diff,
                "price_change_pct": (
                    (price_diff / open_price * 100) if open_price != 0 else 0
                ),
                "candle_body_size": body_size,
                "candle_range": candle_range,
                "body_to_range_ratio": (
                    (body_size / candle_range) if candle_range != 0 else 0
                ),
                "upper_shadow": upper_shadow,
                "lower_shadow": lower_shadow,
                "upper_shadow_ratio": (
                    (upper_shadow / candle_range) if candle_range != 0 else 0
                ),
                "lower_shadow_ratio": (
                    (lower_shadow / candle_range) if candle_range != 0 else 0
                ),
            }

            self.logger.debug(f"Candle features: {features}")
            return features

        except Exception as e:
            self.logger.error(f"Error extracting candle features: {e}")
            raise

    def get_pattern_statistics(self, patterns: List[int]) -> Dict[str, any]:
        """
        Calculate binary pattern distribution statistics.

        Args:
            patterns: List of pattern classifications

        Returns:
            Dictionary with pattern statistics
        """
        if not patterns:
            return {}

        pattern_counts = {}
        for i in range(3):  # Patterns 0-2
            pattern_counts[i] = patterns.count(i)

        total_patterns = len(patterns)

        stats = {
            "total_patterns": total_patterns,
            "pattern_counts": pattern_counts,
            "pattern_percentages": {
                i: (count / total_patterns) * 100 for i, count in pattern_counts.items()
            },
            "pattern_labels": self.classification_config["labels"],
            "bullish_percentage": (pattern_counts.get(1, 0) / total_patterns) * 100,
            "bearish_percentage": (pattern_counts.get(0, 0) / total_patterns) * 100,
            "neutral_percentage": (pattern_counts.get(2, 0) / total_patterns) * 100,
        }

        return stats

    def analyze_market_sentiment(
        self, patterns: List[int], dates: List[datetime]
    ) -> Dict[str, any]:
        """
        Analyze market sentiment over time based on binary patterns.

        Args:
            patterns: List of pattern classifications
            dates: Corresponding dates

        Returns:
            Dictionary with sentiment analysis
        """
        if len(patterns) != len(dates):
            raise ValueError("Patterns and dates must have same length")

        # Create pattern series
        pattern_series = pd.Series(patterns, index=dates)

        # Calculate rolling sentiment (7-day and 30-day windows)
        bullish_signals = (pattern_series == 1).astype(int)
        bearish_signals = (pattern_series == 0).astype(int)

        # Rolling averages
        bullish_7d = bullish_signals.rolling(window=7, min_periods=1).mean()
        bullish_30d = bullish_signals.rolling(window=30, min_periods=1).mean()

        # Streak analysis
        bullish_streaks = self._calculate_streaks(patterns, 1)
        bearish_streaks = self._calculate_streaks(patterns, 0)

        # Monthly sentiment
        monthly_sentiment = self._get_monthly_sentiment(pattern_series)

        sentiment_analysis = {
            "current_sentiment": self._get_current_sentiment(
                patterns[-10:] if len(patterns) >= 10 else patterns
            ),
            "bullish_streaks": bullish_streaks,
            "bearish_streaks": bearish_streaks,
            "monthly_sentiment": monthly_sentiment,
            "trend_strength": {
                "bullish_7d_avg": (
                    float(bullish_7d.iloc[-1]) if len(bullish_7d) > 0 else 0
                ),
                "bullish_30d_avg": (
                    float(bullish_30d.iloc[-1]) if len(bullish_30d) > 0 else 0
                ),
            },
        }

        return sentiment_analysis

    def _calculate_streaks(
        self, patterns: List[int], target_pattern: int
    ) -> Dict[str, any]:
        """Calculate consecutive streaks for a specific pattern."""
        if not patterns:
            return {"max_streak": 0, "current_streak": 0, "total_occurrences": 0}

        streaks = []
        current_streak = 0
        max_streak = 0

        for pattern in patterns:
            if pattern == target_pattern:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0

        # Add final streak if it ends with target pattern
        if current_streak > 0:
            streaks.append(current_streak)

        return {
            "max_streak": max_streak,
            "current_streak": current_streak,
            "total_occurrences": patterns.count(target_pattern),
            "all_streaks": streaks,
            "avg_streak_length": np.mean(streaks) if streaks else 0,
        }

    def _get_current_sentiment(self, recent_patterns: List[int]) -> str:
        """Determine current market sentiment from recent patterns."""
        if not recent_patterns:
            return "Unknown"

        bullish_count = recent_patterns.count(1)
        bearish_count = recent_patterns.count(0)

        if bullish_count > bearish_count * 1.5:
            return "Strong Bullish"
        elif bullish_count > bearish_count:
            return "Bullish"
        elif bearish_count > bullish_count * 1.5:
            return "Strong Bearish"
        elif bearish_count > bullish_count:
            return "Bearish"
        else:
            return "Neutral"

    def _get_monthly_sentiment(
        self, pattern_series: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Get sentiment distribution by month."""
        monthly_sentiment = {}

        for month in range(1, 13):
            month_name = pd.to_datetime(f"2000-{month:02d}-01").strftime("%B")
            monthly_patterns = pattern_series[pattern_series.index.month == month]

            if len(monthly_patterns) > 0:
                bullish_pct = (monthly_patterns == 1).mean() * 100
                bearish_pct = (monthly_patterns == 0).mean() * 100
                neutral_pct = (monthly_patterns == 2).mean() * 100

                monthly_sentiment[month_name] = {
                    "bullish_percentage": bullish_pct,
                    "bearish_percentage": bearish_pct,
                    "neutral_percentage": neutral_pct,
                    "total_days": len(monthly_patterns),
                }
            else:
                monthly_sentiment[month_name] = {
                    "bullish_percentage": 0,
                    "bearish_percentage": 0,
                    "neutral_percentage": 0,
                    "total_days": 0,
                }

        return monthly_sentiment

    def validate_binary_classification(
        self, daily_candle: pd.Series, expected_pattern: int
    ) -> bool:
        """
        Validate binary classification against expected result.

        Args:
            daily_candle: Daily OHLC data
            expected_pattern: Expected classification

        Returns:
            True if classification matches expected
        """
        actual_pattern = self.analyze_binary_pattern(daily_candle)
        return actual_pattern == expected_pattern

    def batch_analyze_patterns(
        self, data: pd.DataFrame, start_date: str = None, end_date: str = None
    ) -> List[Dict]:
        """
        Analyze binary patterns for multiple days in batch.

        Args:
            data: Full daily price data
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            List of binary pattern analysis results
        """
        results = []
        errors = []

        # Get date range for analysis
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = data.index[0]

        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = data.index[-1]

        # Filter analysis dates
        analysis_dates = data.loc[start_dt:end_dt].index

        self.logger.info(
            f"Analyzing binary patterns for {len(analysis_dates)} trading days"
        )

        for i, analysis_date in enumerate(analysis_dates):
            try:
                # Get current day candle
                daily_candle = data.loc[analysis_date]

                # Analyze binary pattern
                pattern = self.analyze_binary_pattern(daily_candle)

                # Extract candle features
                features = self.extract_candle_features(daily_candle)

                # Create result
                result = {
                    "date": analysis_date,
                    "pattern": pattern,
                    "pattern_label": self.classification_config["labels"][pattern],
                    "features": features,
                    "daily_ohlc": {
                        "open": float(daily_candle["Open"]),
                        "high": float(daily_candle["High"]),
                        "low": float(daily_candle["Low"]),
                        "close": float(daily_candle["Close"]),
                    },
                }

                results.append(result)

                if (i + 1) % 100 == 0:
                    self.logger.info(
                        f"Analyzed {i + 1}/{len(analysis_dates)} binary patterns"
                    )

            except Exception as e:
                error_msg = f"Error analyzing {analysis_date}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        self.logger.info(
            f"Binary batch analysis complete: {len(results)} patterns, {len(errors)} errors"
        )

        if errors:
            self.logger.warning(
                f"Errors encountered: {errors[:3]}..."
            )  # Log first 3 errors

        return results


def main():
    """Test the binary pattern analyzer."""
    print("Testing Binary Pattern Analyzer...")

    try:
        # Initialize analyzer
        analyzer = BinaryPatternAnalyzer()

        # Create test data
        test_data = pd.DataFrame(
            {
                "Open": [100.0, 105.0, 102.0, 108.0, 110.0],
                "High": [102.0, 107.0, 104.0, 109.0, 111.0],
                "Low": [99.0, 104.0, 100.0, 107.0, 109.0],
                "Close": [
                    101.0,
                    104.0,
                    103.0,
                    108.0,
                    110.0,
                ],  # Bull, Bear, Bull, Neutral, Neutral
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        print(f"📊 Created test data with {len(test_data)} daily bars")

        # Test single pattern analysis
        test_candle = test_data.iloc[0]  # Bullish candle (100 -> 101)

        pattern = analyzer.analyze_binary_pattern(test_candle)
        features = analyzer.extract_candle_features(test_candle)

        print(f"✅ Single binary pattern analysis:")
        print(f"   📅 Date: {test_data.index[0].date()}")
        print(
            f"   🎯 Pattern: {pattern} ({analyzer.classification_config['labels'][pattern]})"
        )
        print(f"   📊 Features: {features}")

        # Test batch analysis
        results = analyzer.batch_analyze_patterns(test_data)

        print(f"📈 Batch analysis: {len(results)} patterns analyzed")

        # Calculate statistics
        patterns = [r["pattern"] for r in results]
        stats = analyzer.get_pattern_statistics(patterns)

        print(f"📊 Binary Pattern Distribution:")
        for pattern, count in stats["pattern_counts"].items():
            label = stats["pattern_labels"][pattern]
            percentage = stats["pattern_percentages"][pattern]
            print(f"   {pattern}: {label} - {count} ({percentage:.1f}%)")

        print(
            f"📈 Market Sentiment: {stats['bullish_percentage']:.1f}% Bullish, {stats['bearish_percentage']:.1f}% Bearish"
        )

        # Test sentiment analysis
        dates = [r["date"] for r in results]
        sentiment_analysis = analyzer.analyze_market_sentiment(patterns, dates)

        print(f"💹 Current Sentiment: {sentiment_analysis['current_sentiment']}")
        print(
            f"🔥 Max Bullish Streak: {sentiment_analysis['bullish_streaks']['max_streak']}"
        )
        print(
            f"🔻 Max Bearish Streak: {sentiment_analysis['bearish_streaks']['max_streak']}"
        )

        print("✅ Binary pattern analyzer test completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
