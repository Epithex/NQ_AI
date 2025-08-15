#!/usr/bin/env python3
"""
Daily Pattern Analyzer for NQ Trading
Analyzes daily candles against previous day levels for 4-class classification
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import yaml
import logging
import os

class DailyPatternAnalyzer:
    """Analyzes daily candles against previous day levels."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pattern analyzer with configuration."""
        self.config = self.load_config(config_path)
        self.classification_config = self.config['classification']
        self.feature_config = self.config['features']
        self.setup_logging()
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging for the analyzer."""
        log_dir = self.config['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(f"{log_dir}/pattern_analyzer.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_daily_pattern(self, daily_candle: pd.Series, prev_high: float, prev_low: float) -> int:
        """
        Classify daily pattern based on previous day level interaction.
        
        4-class classification system:
        1: High Breakout - Daily high >= previous day high only
        2: Low Breakdown - Daily low <= previous day low only  
        3: Range Expansion - Both levels touched
        4: Range Bound - Neither level touched
        
        Args:
            daily_candle: Daily OHLC data
            prev_high: Previous day high
            prev_low: Previous day low
            
        Returns:
            Pattern classification (1-4)
        """
        try:
            # Extract daily high and low
            daily_high = float(daily_candle['High'])
            daily_low = float(daily_candle['Low'])
            
            # Check level interactions
            took_high = daily_high >= prev_high
            took_low = daily_low <= prev_low
            
            # Classify based on touch pattern
            if took_high and not took_low:
                pattern = 1  # High breakout only
                description = "High Breakout"
            elif took_low and not took_high:
                pattern = 2  # Low breakdown only
                description = "Low Breakdown"
            elif took_high and took_low:
                pattern = 3  # Range expansion (both levels)
                description = "Range Expansion"
            else:
                pattern = 4  # Range bound (neither level)
                description = "Range Bound"
            
            self.logger.debug(f"Pattern {pattern}: {description} | High: {took_high}, Low: {took_low}")
            self.logger.debug(f"Daily H/L: {daily_high:.2f}/{daily_low:.2f} vs Prev H/L: {prev_high:.2f}/{prev_low:.2f}")
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern: {e}")
            raise
    
    def extract_numerical_features(self, daily_candle: pd.Series, prev_high: float, prev_low: float) -> Dict[str, float]:
        """
        Extract distance-based numerical features.
        
        Args:
            daily_candle: Daily OHLC data
            prev_high: Previous day high
            prev_low: Previous day low
            
        Returns:
            Dictionary of numerical features
        """
        try:
            open_price = float(daily_candle['Open'])
            
            features = {
                'distance_to_prev_high': prev_high - open_price,  # Distance from open to prev high
                'distance_to_prev_low': open_price - prev_low,    # Distance from open to prev low
                'prev_day_range': prev_high - prev_low            # Size of previous day's range
            }
            
            self.logger.debug(f"Features: {features}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise
    
    def get_pattern_statistics(self, patterns: List[int]) -> Dict[str, any]:
        """
        Calculate pattern distribution statistics.
        
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
            'total_patterns': total_patterns,
            'pattern_counts': pattern_counts,
            'pattern_percentages': {
                i: (count / total_patterns) * 100 
                for i, count in pattern_counts.items()
            },
            'pattern_labels': self.classification_config['labels']
        }
        
        return stats
    
    def analyze_pattern_sequences(self, patterns: List[int], dates: List[datetime]) -> Dict[str, any]:
        """
        Analyze patterns over time sequences.
        
        Args:
            patterns: List of pattern classifications
            dates: Corresponding dates
            
        Returns:
            Dictionary with sequence analysis
        """
        if len(patterns) != len(dates):
            raise ValueError("Patterns and dates must have same length")
        
        # Create pattern series
        pattern_series = pd.Series(patterns, index=dates)
        
        # Calculate consecutive patterns
        consecutive_analysis = {}
        for pattern in range(1, 5):
            consecutive_analysis[pattern] = self._find_consecutive_patterns(pattern_series, pattern)
        
        # Calculate pattern transitions
        transitions = self._calculate_pattern_transitions(patterns)
        
        # Monthly pattern distribution
        monthly_dist = self._get_monthly_pattern_distribution(pattern_series)
        
        sequence_analysis = {
            'consecutive_patterns': consecutive_analysis,
            'pattern_transitions': transitions,
            'monthly_distribution': monthly_dist,
            'most_common_pattern': max(patterns, key=patterns.count),
            'least_common_pattern': min(patterns, key=patterns.count)
        }
        
        return sequence_analysis
    
    def _find_consecutive_patterns(self, pattern_series: pd.Series, target_pattern: int) -> Dict[str, any]:
        """Find consecutive occurrences of a specific pattern."""
        is_target = pattern_series == target_pattern
        
        # Find consecutive groups
        groups = (is_target != is_target.shift()).cumsum()
        consecutive_groups = pattern_series[is_target].groupby(groups[is_target])
        
        if consecutive_groups.ngroups == 0:
            return {'max_consecutive': 0, 'total_occurrences': 0, 'sequences': []}
        
        sequence_lengths = consecutive_groups.size()
        
        return {
            'max_consecutive': sequence_lengths.max(),
            'total_occurrences': len(pattern_series[pattern_series == target_pattern]),
            'sequences': sequence_lengths.tolist()
        }
    
    def _calculate_pattern_transitions(self, patterns: List[int]) -> Dict[str, Dict[str, int]]:
        """Calculate transition matrix between patterns."""
        transitions = {}
        
        for from_pattern in range(1, 5):
            transitions[from_pattern] = {}
            for to_pattern in range(1, 5):
                transitions[from_pattern][to_pattern] = 0
        
        # Count transitions
        for i in range(len(patterns) - 1):
            from_pattern = patterns[i]
            to_pattern = patterns[i + 1]
            transitions[from_pattern][to_pattern] += 1
        
        return transitions
    
    def _get_monthly_pattern_distribution(self, pattern_series: pd.Series) -> Dict[str, Dict[int, int]]:
        """Get pattern distribution by month."""
        monthly_dist = {}
        
        for month in range(1, 13):
            month_name = pd.to_datetime(f'2000-{month:02d}-01').strftime('%B')
            monthly_patterns = pattern_series[pattern_series.index.month == month]
            
            month_counts = {}
            for pattern in range(1, 5):
                month_counts[pattern] = (monthly_patterns == pattern).sum()
            
            monthly_dist[month_name] = month_counts
        
        return monthly_dist
    
    def validate_pattern_classification(self, daily_candle: pd.Series, prev_high: float, 
                                      prev_low: float, expected_pattern: int) -> bool:
        """
        Validate pattern classification against expected result.
        
        Args:
            daily_candle: Daily OHLC data
            prev_high: Previous day high
            prev_low: Previous day low
            expected_pattern: Expected classification
            
        Returns:
            True if classification matches expected
        """
        actual_pattern = self.analyze_daily_pattern(daily_candle, prev_high, prev_low)
        return actual_pattern == expected_pattern
    
    def batch_analyze_patterns(self, data: pd.DataFrame, start_date: str = None, 
                              end_date: str = None) -> List[Dict]:
        """
        Analyze patterns for multiple days in batch.
        
        Args:
            data: Full daily price data
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of pattern analysis results
        """
        results = []
        errors = []
        
        # Get date range for analysis
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = data.index[1]  # Start after first day (need previous day)
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = data.index[-1]
        
        # Filter analysis dates
        analysis_dates = data.loc[start_dt:end_dt].index
        
        self.logger.info(f"Analyzing patterns for {len(analysis_dates)} trading days")
        
        for i, analysis_date in enumerate(analysis_dates):
            try:
                # Get current day candle
                daily_candle = data.loc[analysis_date]
                
                # Get previous day levels
                prev_idx = data.index.get_indexer([analysis_date])[0] - 1
                if prev_idx < 0:
                    continue
                
                prev_day = data.iloc[prev_idx]
                prev_high = float(prev_day['High'])
                prev_low = float(prev_day['Low'])
                
                # Analyze pattern
                pattern = self.analyze_daily_pattern(daily_candle, prev_high, prev_low)
                
                # Extract features
                features = self.extract_numerical_features(daily_candle, prev_high, prev_low)
                
                # Create result
                result = {
                    'date': analysis_date,
                    'pattern': pattern,
                    'pattern_label': self.classification_config['labels'][pattern],
                    'features': features,
                    'prev_high': prev_high,
                    'prev_low': prev_low,
                    'daily_ohlc': {
                        'open': float(daily_candle['Open']),
                        'high': float(daily_candle['High']),
                        'low': float(daily_candle['Low']),
                        'close': float(daily_candle['Close'])
                    }
                }
                
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Analyzed {i + 1}/{len(analysis_dates)} patterns")
                
            except Exception as e:
                error_msg = f"Error analyzing {analysis_date}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        self.logger.info(f"Batch analysis complete: {len(results)} patterns, {len(errors)} errors")
        
        if errors:
            self.logger.warning(f"Errors encountered: {errors[:3]}...")  # Log first 3 errors
        
        return results

def main():
    """Test the daily pattern analyzer."""
    print("Testing Daily Pattern Analyzer...")
    
    try:
        # Initialize analyzer
        analyzer = DailyPatternAnalyzer()
        
        # Load test data
        test_data_path = "data/metadata/test_nq_data.csv"
        
        if not os.path.exists(test_data_path):
            print("❌ Test data not found. Run stooq_fetcher.py first.")
            return 1
        
        # Load data
        data = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
        print(f"📊 Loaded {len(data)} daily bars")
        
        # Test single pattern analysis
        test_date = data.index[-10]  # 10 days from end
        daily_candle = data.loc[test_date]
        
        # Get previous day levels
        prev_idx = data.index.get_indexer([test_date])[0] - 1
        prev_day = data.iloc[prev_idx]
        prev_high = float(prev_day['High'])
        prev_low = float(prev_day['Low'])
        
        # Analyze pattern
        pattern = analyzer.analyze_daily_pattern(daily_candle, prev_high, prev_low)
        features = analyzer.extract_numerical_features(daily_candle, prev_high, prev_low)
        
        print(f"✅ Single pattern analysis:")
        print(f"   📅 Date: {test_date.date()}")
        print(f"   🎯 Pattern: {pattern} ({analyzer.classification_config['labels'][pattern]})")
        print(f"   📊 Features: {features}")
        
        # Test batch analysis (last 50 days)
        recent_start = data.index[-50]
        results = analyzer.batch_analyze_patterns(
            data=data,
            start_date=recent_start.strftime('%Y-%m-%d')
        )
        
        print(f"📈 Batch analysis: {len(results)} patterns analyzed")
        
        # Calculate statistics
        patterns = [r['pattern'] for r in results]
        stats = analyzer.get_pattern_statistics(patterns)
        
        print(f"📊 Pattern Distribution:")
        for pattern, count in stats['pattern_counts'].items():
            label = stats['pattern_labels'][pattern]
            percentage = stats['pattern_percentages'][pattern]
            print(f"   {pattern}: {label} - {count} ({percentage:.1f}%)")
        
        # Test sequence analysis
        dates = [r['date'] for r in results]
        sequence_analysis = analyzer.analyze_pattern_sequences(patterns, dates)
        
        print(f"🔄 Most common pattern: {sequence_analysis['most_common_pattern']}")
        print(f"🔄 Least common pattern: {sequence_analysis['least_common_pattern']}")
        
        print("✅ Pattern analyzer test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())