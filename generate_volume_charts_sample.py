#!/usr/bin/env python3
"""
Generate 10 Days of NASDAQ Charts with Volume Bars
Quick test script to generate sample charts with volume included
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from daily_chart_generator import DailyChartGenerator
from excel_data_fetcher import ExcelDataFetcher

def main():
    print("🚀 Generating 10 Days of NASDAQ Charts with Volume Bars")
    
    try:
        # Initialize data fetcher and chart generator
        data_fetcher = ExcelDataFetcher()
        chart_generator = DailyChartGenerator()
        
        print("📊 Loading NASDAQ data...")
        
        # Fetch NASDAQ data (most recent available)
        nasdaq_data = data_fetcher.fetch_instrument_data("NASDAQ", 2024, 2025)
        
        if nasdaq_data is None or len(nasdaq_data) == 0:
            print("❌ No NASDAQ data available")
            return 1
            
        print(f"✅ Loaded {len(nasdaq_data)} NASDAQ bars")
        print(f"   Date range: {nasdaq_data.index[0].date()} to {nasdaq_data.index[-1].date()}")
        
        # Ensure volume column exists and has valid data
        if 'Volume' not in nasdaq_data.columns:
            print("⚠️  Volume column missing, adding synthetic volume...")
            # Generate synthetic volume if missing
            nasdaq_data['Volume'] = np.random.lognormal(15, 0.5, len(nasdaq_data)).astype(int)
        
        # Get the last 11 days (10 for analysis + 1 for previous day reference)
        if len(nasdaq_data) < 31:  # Need at least 31 days (30 for chart + 1 for analysis)
            print("❌ Not enough data for chart generation (need at least 31 days)")
            return 1
            
        # Select last 10 trading days for analysis
        analysis_dates = nasdaq_data.index[-10:]  # Last 10 days
        
        print(f"📈 Generating charts for dates: {analysis_dates[0].date()} to {analysis_dates[-1].date()}")
        
        # Create output directory for volume charts
        volume_charts_dir = Path("data/volume_chart_samples")
        volume_charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Update chart generator config to use our sample directory
        chart_generator.config["paths"]["images"] = str(volume_charts_dir)
        
        generated_charts = []
        
        # Generate charts for each of the 10 days
        for i, analysis_date in enumerate(analysis_dates, 1):
            try:
                print(f"🎨 Generating chart {i}/10 for {analysis_date.date()}...")
                
                # Get previous day data for reference lines
                analysis_idx = nasdaq_data.index.get_indexer([analysis_date])[0]
                previous_candle = nasdaq_data.iloc[analysis_idx - 1]
                
                # Generate chart with volume bars
                chart_path = chart_generator.generate_daily_chart_image(
                    analysis_date=analysis_date,
                    data=nasdaq_data,
                    previous_candle=previous_candle,
                    instrument="NASDAQ"
                )
                
                generated_charts.append(chart_path)
                
                print(f"   ✅ Saved: {Path(chart_path).name}")
                
                # Show some stats
                current_candle = nasdaq_data.loc[analysis_date]
                print(f"   📊 OHLC: ${current_candle['Open']:.2f} / ${current_candle['High']:.2f} / ${current_candle['Low']:.2f} / ${current_candle['Close']:.2f}")
                print(f"   📈 Volume: {current_candle['Volume']:,}")
                print(f"   📋 Prev levels - H: ${previous_candle['High']:.2f}, L: ${previous_candle['Low']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error generating chart for {analysis_date.date()}: {e}")
                continue
        
        print(f"\n✅ Chart generation complete!")
        print(f"   📁 Generated {len(generated_charts)} charts with volume bars")
        print(f"   📂 Location: {volume_charts_dir}")
        print(f"   📏 Size: 224x224 pixels each")
        
        # Show total file size
        total_size = sum(os.path.getsize(chart) for chart in generated_charts if os.path.exists(chart))
        print(f"   💾 Total size: {total_size / 1024:.1f} KB")
        
        # List generated files
        print(f"\n📋 Generated charts:")
        for chart_path in generated_charts:
            if os.path.exists(chart_path):
                filename = Path(chart_path).name
                size_kb = os.path.getsize(chart_path) / 1024
                print(f"   • {filename} ({size_kb:.1f} KB)")
        
        print(f"\n🎯 Volume Chart Features:")
        print(f"   • 30-bar candlestick charts (6 weeks context)")
        print(f"   • Volume bars below price action (3:1 panel ratio)")
        print(f"   • Previous day high/low reference lines")
        print(f"   • Green line: Previous day high")
        print(f"   • Red line: Previous day low") 
        print(f"   • 224x224 resolution for ViT input")
        print(f"   • Enhanced visual context for pattern analysis")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())