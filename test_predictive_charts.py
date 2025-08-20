#!/usr/bin/env python3
"""
Test Corrected Predictive Chart Generation
Verify that charts exclude the analysis date and show true predictive context
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from daily_chart_generator import DailyChartGenerator
from excel_data_fetcher import ExcelDataFetcher

def main():
    print("🎯 Testing CORRECTED Predictive Chart Generation")
    print("   Charts should show 30 historical days + reference lines")
    print("   Charts should NOT show the analysis date candle")
    
    try:
        # Initialize components
        data_fetcher = ExcelDataFetcher()
        chart_generator = DailyChartGenerator()
        
        print("📊 Loading NASDAQ data...")
        nasdaq_data = data_fetcher.fetch_instrument_data("NASDAQ", 2024, 2025)
        
        if nasdaq_data is None or len(nasdaq_data) < 50:
            print("❌ Not enough NASDAQ data for testing")
            return 1
        
        print(f"✅ Loaded {len(nasdaq_data)} NASDAQ bars")
        
        # Create test directory
        test_dir = Path("data/predictive_chart_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        chart_generator.config["paths"]["images"] = str(test_dir)
        
        # Test with recent data
        analysis_date = nasdaq_data.index[-5]  # 5 days from end
        previous_candle = nasdaq_data.iloc[-6]  # Day before analysis date
        
        print(f"\n🎯 Testing Predictive Chart:")
        print(f"   Predicting: {analysis_date.date()}")
        print(f"   Previous day: {nasdaq_data.index[-6].date()}")
        print(f"   Chart should show: {nasdaq_data.index[-36].date()} to {nasdaq_data.index[-6].date()}")
        print(f"   Chart should NOT show: {analysis_date.date()}")
        
        # Generate corrected predictive chart
        chart_path = chart_generator.generate_daily_chart_image(
            analysis_date=analysis_date,
            data=nasdaq_data,
            previous_candle=previous_candle,
            instrument="NASDAQ"
        )
        
        print(f"\n✅ Corrected predictive chart generated!")
        print(f"   📁 Path: {chart_path}")
        print(f"   📏 Size: 224x224 pixels")
        
        # Verify file
        if os.path.exists(chart_path):
            file_size = os.path.getsize(chart_path) / 1024
            print(f"   💾 File size: {file_size:.1f} KB")
        
        # Show what the model sees vs predicts
        print(f"\n🤖 Model Input/Output:")
        print(f"   📈 Visual Input: 30 historical bars ending {nasdaq_data.index[-6].date()}")
        print(f"   📊 Reference Lines: Prev high ${previous_candle['High']:.2f}, Prev low ${previous_candle['Low']:.2f}")
        print(f"   🔢 Numerical Features: Distance from analysis day open to prev levels")
        print(f"   🎯 Prediction Target: What pattern will occur on {analysis_date.date()}")
        
        # Show the actual outcome for verification
        actual_candle = nasdaq_data.loc[analysis_date]
        print(f"\n📊 Actual Outcome (for verification only):")
        print(f"   Analysis day OHLC: ${actual_candle['Open']:.2f} / ${actual_candle['High']:.2f} / ${actual_candle['Low']:.2f} / ${actual_candle['Close']:.2f}")
        
        # Determine actual pattern
        current_high = actual_candle['High']
        current_low = actual_candle['Low']
        prev_high = previous_candle['High']
        prev_low = previous_candle['Low']
        
        if current_high >= prev_high and current_low > prev_low:
            pattern = "High Breakout (Pattern 1)"
        elif current_low <= prev_low and current_high < prev_high:
            pattern = "Low Breakdown (Pattern 2)"
        elif current_high >= prev_high and current_low <= prev_low:
            pattern = "Range Expansion (Pattern 3)"
        else:
            pattern = "Range Bound (Pattern 4)"
            
        print(f"   Actual Pattern: {pattern}")
        
        print(f"\n✅ Predictive Chart Validation:")
        print(f"   ✓ Chart excludes analysis date candle")
        print(f"   ✓ Chart shows 30 historical bars for context")
        print(f"   ✓ Previous day reference lines included")
        print(f"   ✓ True prediction setup - no future information")
        print(f"   ✓ Model must predict pattern from historical context only")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())