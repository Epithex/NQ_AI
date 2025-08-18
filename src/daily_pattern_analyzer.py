#!/usr/bin/env python3
"""
Daily Pattern Analyzer for NQ Trading
Analyzes daily candles for 4-class previous day levels classification
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import yaml
import logging
import os


class DailyPatternAnalyzer:
    """Analyzes daily candles for 4-class previous day levels classification."""

    def __init__(self, config_path: str = "config/config_daily_hybrid.yaml"):
        """Initialize daily pattern analyzer with configuration."""
        self.config = self.load_config(config_path)
        self.classification_config = self.config["classification"]
        self.features_config = self.config["features"]
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
                logging.FileHandler(f"{log_dir}/daily_pattern_analyzer.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def analyze_daily_pattern(self, current_candle: pd.Series, previous_candle: pd.Series) -> int:
        """
        Classify daily pattern based on previous day level interactions.

        4-class previous day levels classification system:
        1: High Breakout - Daily high >= prev_high AND daily low > prev_low
        2: Low Breakdown - Daily low <= prev_low AND daily high < prev_high  
        3: Range Expansion - Daily high >= prev_high AND daily low <= prev_low
        4: Range Bound - Daily high < prev_high AND daily low > prev_low

        Args:
            current_candle: Current day OHLC data
            previous_candle: Previous day OHLC data

        Returns:
            Pattern classification (1-4)
        """
        try:
            # Extract current day prices
            current_high = float(current_candle["High"])
            current_low = float(current_candle["Low"])
            current_open = float(current_candle["Open"])
            current_close = float(current_candle["Close"])

            # Extract previous day levels
            prev_high = float(previous_candle["High"])
            prev_low = float(previous_candle["Low"])

            # Apply 4-class classification logic
            if current_high >= prev_high and current_low > prev_low:
                pattern = 1  # High Breakout
                description = "High Breakout"
            elif current_low <= prev_low and current_high < prev_high:
                pattern = 2  # Low Breakdown
                description = "Low Breakdown"
            elif current_high >= prev_high and current_low <= prev_low:
                pattern = 3  # Range Expansion
                description = "Range Expansion"
            else:  # current_high < prev_high and current_low > prev_low
                pattern = 4  # Range Bound
                description = "Range Bound"

            # Calculate additional metrics for logging
            range_expansion = (current_high - current_low) - (prev_high - prev_low)
            prev_high_penetration = current_high - prev_high
            prev_low_penetration = prev_low - current_low

            self.logger.debug(f"Pattern {pattern}: {description}")
            self.logger.debug(f"Current: H={current_high:.2f}, L={current_low:.2f}")
            self.logger.debug(f"Previous: H={prev_high:.2f}, L={prev_low:.2f}")
            self.logger.debug(f"Range expansion: {range_expansion:.2f}")
            self.logger.debug(f"High penetration: {prev_high_penetration:.2f}")
            self.logger.debug(f"Low penetration: {prev_low_penetration:.2f}")

            return pattern

        except Exception as e:
            self.logger.error(f"Error analyzing daily pattern: {e}")
            raise

    def extract_numerical_features(self, current_candle: pd.Series, previous_candle: pd.Series) -> List[float]:
        """
        Extract the 3 numerical features for hybrid model training.

        Features:
        1. Distance to previous high (prev_high - open_price)
        2. Distance to previous low (open_price - prev_low)  
        3. Previous day range (prev_high - prev_low)

        Args:
            current_candle: Current day OHLC data
            previous_candle: Previous day OHLC data

        Returns:
            List of 3 numerical features
        """
        try:
            # Extract prices
            current_open = float(current_candle["Open"])
            prev_high = float(previous_candle["High"])
            prev_low = float(previous_candle["Low"])

            # Calculate the 3 numerical features
            distance_to_prev_high = prev_high - current_open
            distance_to_prev_low = current_open - prev_low
            prev_day_range = prev_high - prev_low

            features = [distance_to_prev_high, distance_to_prev_low, prev_day_range]

            self.logger.debug(f"Numerical features: {features}")
            self.logger.debug(f"Distance to prev high: {distance_to_prev_high:.2f}")
            self.logger.debug(f"Distance to prev low: {distance_to_prev_low:.2f}")
            self.logger.debug(f"Previous day range: {prev_day_range:.2f}")

            return features

        except Exception as e:
            self.logger.error(f"Error extracting numerical features: {e}")
            raise

    def extract_candle_features(self, current_candle: pd.Series, previous_candle: pd.Series) -> Dict[str, float]:
        """
        Extract comprehensive candle-based features for analysis.

        Args:
            current_candle: Current day OHLC data
            previous_candle: Previous day OHLC data

        Returns:
            Dictionary of candle features including numerical features
        """
        try:
            # Current day prices
            current_open = float(current_candle["Open"])
            current_high = float(current_candle["High"])
            current_low = float(current_candle["Low"])
            current_close = float(current_candle["Close"])

            # Previous day levels
            prev_high = float(previous_candle["High"])
            prev_low = float(previous_candle["Low"])
            prev_close = float(previous_candle["Close"])

            # Basic candle metrics
            price_diff = current_close - current_open
            current_range = current_high - current_low
            body_size = abs(price_diff)

            # Previous day level interactions
            high_vs_prev_high = current_high - prev_high
            low_vs_prev_low = current_low - prev_low
            range_expansion = current_range - (prev_high - prev_low)

            # Gap analysis
            gap_from_prev_close = current_open - prev_close

            # The 3 numerical features for the model
            numerical_features = self.extract_numerical_features(current_candle, previous_candle)

            features = {
                # Basic candle features
                "price_change": price_diff,
                "price_change_pct": (price_diff / current_open * 100) if current_open != 0 else 0,
                "candle_body_size": body_size,
                "current_range": current_range,
                "body_to_range_ratio": (body_size / current_range) if current_range != 0 else 0,
                
                # Previous day level interactions
                "high_vs_prev_high": high_vs_prev_high,
                "low_vs_prev_low": low_vs_prev_low,
                "range_expansion": range_expansion,
                "gap_from_prev_close": gap_from_prev_close,
                
                # The 3 numerical features for the model
                "distance_to_prev_high": numerical_features[0],
                "distance_to_prev_low": numerical_features[1],
                "prev_day_range": numerical_features[2],
                
                # Penetration ratios
                "high_penetration_ratio": (high_vs_prev_high / (prev_high - prev_low)) if (prev_high - prev_low) != 0 else 0,
                "low_penetration_ratio": (abs(low_vs_prev_low) / (prev_high - prev_low)) if (prev_high - prev_low) != 0 else 0,
            }

            self.logger.debug(f"Comprehensive candle features: {features}")
            return features

        except Exception as e:
            self.logger.error(f"Error extracting candle features: {e}")
            raise

    def get_pattern_statistics(self, patterns: List[int]) -> Dict[str, any]:
        """
        Calculate 4-class pattern distribution statistics.

        Args:
            patterns: List of pattern classifications

        Returns:
            Dictionary with pattern statistics
        """
        if not patterns:
            return {}

        pattern_counts = {}
        for i in range(1, 5):  # Patterns 1-4
            pattern_counts[i] = patterns.count(i)

        total_patterns = len(patterns)

        stats = {
            "total_patterns": total_patterns,
            "pattern_counts": pattern_counts,
            "pattern_percentages": {
                i: (count / total_patterns) * 100 for i, count in pattern_counts.items()
            },
            "pattern_labels": self.classification_config["labels"],
            "high_breakout_percentage": (pattern_counts.get(1, 0) / total_patterns) * 100,
            "low_breakdown_percentage": (pattern_counts.get(2, 0) / total_patterns) * 100,
            "range_expansion_percentage": (pattern_counts.get(3, 0) / total_patterns) * 100,
            "range_bound_percentage": (pattern_counts.get(4, 0) / total_patterns) * 100,
        }

        return stats

    def analyze_market_behavior(
        self, patterns: List[int], dates: List[datetime]
    ) -> Dict[str, any]:
        """
        Analyze market behavior over time based on 4-class patterns.

        Args:
            patterns: List of pattern classifications
            dates: Corresponding dates

        Returns:
            Dictionary with market behavior analysis
        """
        if len(patterns) != len(dates):
            raise ValueError("Patterns and dates must have same length")

        # Create pattern series
        pattern_series = pd.Series(patterns, index=dates)

        # Calculate pattern frequencies
        breakout_signals = (pattern_series == 1).astype(int)  # High Breakout
        breakdown_signals = (pattern_series == 2).astype(int)  # Low Breakdown
        expansion_signals = (pattern_series == 3).astype(int)  # Range Expansion
        range_bound_signals = (pattern_series == 4).astype(int)  # Range Bound

        # Rolling averages (7-day and 30-day windows)
        breakout_7d = breakout_signals.rolling(window=7, min_periods=1).mean()
        breakdown_7d = breakdown_signals.rolling(window=7, min_periods=1).mean()
        expansion_7d = expansion_signals.rolling(window=7, min_periods=1).mean()
        range_bound_7d = range_bound_signals.rolling(window=7, min_periods=1).mean()

        # Streak analysis for each pattern
        breakout_streaks = self._calculate_streaks(patterns, 1)
        breakdown_streaks = self._calculate_streaks(patterns, 2)
        expansion_streaks = self._calculate_streaks(patterns, 3)
        range_bound_streaks = self._calculate_streaks(patterns, 4)

        # Market regime analysis
        market_regime = self._get_market_regime(
            patterns[-10:] if len(patterns) >= 10 else patterns
        )

        # Monthly pattern distribution
        monthly_patterns = self._get_monthly_patterns(pattern_series)

        behavior_analysis = {
            "market_regime": market_regime,
            "pattern_streaks": {
                "high_breakout": breakout_streaks,
                "low_breakdown": breakdown_streaks,
                "range_expansion": expansion_streaks,
                "range_bound": range_bound_streaks,
            },
            "monthly_patterns": monthly_patterns,
            "trend_strength": {
                "breakout_7d_avg": float(breakout_7d.iloc[-1]) if len(breakout_7d) > 0 else 0,
                "breakdown_7d_avg": float(breakdown_7d.iloc[-1]) if len(breakdown_7d) > 0 else 0,
                "expansion_7d_avg": float(expansion_7d.iloc[-1]) if len(expansion_7d) > 0 else 0,
                "range_bound_7d_avg": float(range_bound_7d.iloc[-1]) if len(range_bound_7d) > 0 else 0,
            },
        }

        return behavior_analysis

    def _calculate_streaks(self, patterns: List[int], target_pattern: int) -> Dict[str, any]:
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

    def _get_market_regime(self, recent_patterns: List[int]) -> str:
        """Determine current market regime from recent patterns."""
        if not recent_patterns:
            return "Unknown"

        breakout_count = recent_patterns.count(1)
        breakdown_count = recent_patterns.count(2)
        expansion_count = recent_patterns.count(3)
        range_bound_count = recent_patterns.count(4)

        total = len(recent_patterns)
        
        # Determine dominant pattern
        max_count = max(breakout_count, breakdown_count, expansion_count, range_bound_count)
        
        if max_count / total >= 0.5:  # Strong regime (50%+ of one pattern)
            if breakout_count == max_count:
                return "Strong Breakout Trend"
            elif breakdown_count == max_count:
                return "Strong Breakdown Trend"
            elif expansion_count == max_count:
                return "High Volatility"
            else:
                return "Range Bound"
        elif (breakout_count + breakdown_count) / total >= 0.6:
            return "Trending Market"
        elif (expansion_count + range_bound_count) / total >= 0.6:
            return "Consolidating Market"
        else:
            return "Mixed Regime"

    def _get_monthly_patterns(self, pattern_series: pd.Series) -> Dict[str, Dict[str, float]]:
        """Get pattern distribution by month."""
        monthly_patterns = {}

        for month in range(1, 13):
            month_name = pd.to_datetime(f"2000-{month:02d}-01").strftime("%B")
            month_data = pattern_series[pattern_series.index.month == month]

            if len(month_data) > 0:
                breakout_pct = (month_data == 1).mean() * 100
                breakdown_pct = (month_data == 2).mean() * 100
                expansion_pct = (month_data == 3).mean() * 100
                range_bound_pct = (month_data == 4).mean() * 100

                monthly_patterns[month_name] = {
                    "high_breakout_percentage": breakout_pct,
                    "low_breakdown_percentage": breakdown_pct,
                    "range_expansion_percentage": expansion_pct,
                    "range_bound_percentage": range_bound_pct,
                    "total_days": len(month_data),
                }
            else:
                monthly_patterns[month_name] = {
                    "high_breakout_percentage": 0,
                    "low_breakdown_percentage": 0,
                    "range_expansion_percentage": 0,
                    "range_bound_percentage": 0,
                    "total_days": 0,
                }

        return monthly_patterns

    def validate_daily_classification(
        self, current_candle: pd.Series, previous_candle: pd.Series, expected_pattern: int
    ) -> bool:
        """
        Validate daily classification against expected result.

        Args:
            current_candle: Current day OHLC data
            previous_candle: Previous day OHLC data
            expected_pattern: Expected classification

        Returns:
            True if classification matches expected
        """
        actual_pattern = self.analyze_daily_pattern(current_candle, previous_candle)
        return actual_pattern == expected_pattern

    def batch_analyze_patterns(
        self, data: pd.DataFrame, start_date: str = None, end_date: str = None
    ) -> List[Dict]:
        """
        Analyze daily patterns for multiple days in batch.

        Args:
            data: Full daily price data
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            List of daily pattern analysis results
        """
        results = []
        errors = []

        # Get date range for analysis - need to start from second day (need previous day)
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = data.index[1]  # Start from second day

        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = data.index[-1]

        # Filter analysis dates (skip first day as we need previous day data)
        analysis_dates = data.loc[start_dt:end_dt].index

        self.logger.info(
            f"Analyzing daily patterns for {len(analysis_dates)} trading days"
        )

        for i, analysis_date in enumerate(analysis_dates):
            try:
                # Get current day candle
                current_candle = data.loc[analysis_date]
                
                # Get previous day candle
                current_index = data.index.get_loc(analysis_date)
                if current_index == 0:
                    continue  # Skip first day
                previous_candle = data.iloc[current_index - 1]

                # Analyze daily pattern
                pattern = self.analyze_daily_pattern(current_candle, previous_candle)

                # Extract comprehensive features
                features = self.extract_candle_features(current_candle, previous_candle)
                
                # Extract numerical features for the model
                numerical_features = self.extract_numerical_features(current_candle, previous_candle)

                # Create result
                result = {
                    "date": analysis_date,
                    "pattern": pattern,
                    "pattern_label": self.classification_config["labels"][pattern],
                    "features": features,
                    "numerical_features": numerical_features,
                    "current_ohlc": {
                        "open": float(current_candle["Open"]),
                        "high": float(current_candle["High"]),
                        "low": float(current_candle["Low"]),
                        "close": float(current_candle["Close"]),
                    },
                    "previous_levels": {
                        "high": float(previous_candle["High"]),
                        "low": float(previous_candle["Low"]),
                    },
                }

                results.append(result)

                if (i + 1) % 100 == 0:
                    self.logger.info(
                        f"Analyzed {i + 1}/{len(analysis_dates)} daily patterns"
                    )

            except Exception as e:
                error_msg = f"Error analyzing {analysis_date}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        self.logger.info(
            f"Daily batch analysis complete: {len(results)} patterns, {len(errors)} errors"
        )

        if errors:
            self.logger.warning(
                f"Errors encountered: {errors[:3]}..."
            )  # Log first 3 errors

        return results


def main():
    """Test the daily pattern analyzer."""
    print("Testing Daily Pattern Analyzer...")

    try:
        # Initialize analyzer
        analyzer = DailyPatternAnalyzer()

        # Create test data with previous day context
        test_data = pd.DataFrame(
            {
                "Open": [100.0, 105.0, 102.0, 108.0, 110.0, 109.0],
                "High": [102.0, 107.0, 104.0, 109.0, 112.0, 111.0],
                "Low": [99.0, 104.0, 100.0, 107.0, 109.0, 108.0],
                "Close": [101.0, 104.0, 103.0, 108.5, 111.0, 110.0],
            },
            index=pd.date_range("2023-01-01", periods=6),
        )

        print(f"📊 Created test data with {len(test_data)} daily bars")

        # Test single pattern analysis (need current + previous day)
        current_candle = test_data.iloc[1]  # Second day
        previous_candle = test_data.iloc[0]  # First day

        pattern = analyzer.analyze_daily_pattern(current_candle, previous_candle)
        features = analyzer.extract_candle_features(current_candle, previous_candle)
        numerical_features = analyzer.extract_numerical_features(current_candle, previous_candle)

        print(f"✅ Single daily pattern analysis:")
        print(f"   📅 Date: {test_data.index[1].date()}")
        print(f"   🎯 Pattern: {pattern} ({analyzer.classification_config['labels'][pattern]})")
        print(f"   📊 Numerical Features: {numerical_features}")
        print(f"   📈 All Features: {features}")

        # Test batch analysis
        results = analyzer.batch_analyze_patterns(test_data)

        print(f"📈 Batch analysis: {len(results)} patterns analyzed")

        # Calculate statistics
        patterns = [r["pattern"] for r in results]
        stats = analyzer.get_pattern_statistics(patterns)

        print(f"📊 Daily Pattern Distribution:")
        for pattern, count in stats["pattern_counts"].items():
            label = stats["pattern_labels"][pattern]
            percentage = stats["pattern_percentages"][pattern]
            print(f"   {pattern}: {label} - {count} ({percentage:.1f}%)")

        # Test market behavior analysis
        dates = [r["date"] for r in results]
        behavior_analysis = analyzer.analyze_market_behavior(patterns, dates)

        print(f"💹 Market Regime: {behavior_analysis['market_regime']}")
        print(f"🚀 Max Breakout Streak: {behavior_analysis['pattern_streaks']['high_breakout']['max_streak']}")
        print(f"📉 Max Breakdown Streak: {behavior_analysis['pattern_streaks']['low_breakdown']['max_streak']}")

        print("✅ Daily pattern analyzer test completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
