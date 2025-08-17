#!/usr/bin/env python3
"""
Excel Data Structure Analyzer
Analyzes the consolidated futures data Excel file to understand data organization
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def analyze_excel_structure():
    """Analyze the Excel file structure and content."""
    excel_file = "data/Futures_Data_Consolidated.xlsx"
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return None
    
    print("🔍 Analyzing Excel file structure...")
    print(f"📁 File: {excel_file}")
    print("=" * 60)
    
    try:
        # Load Excel file and examine sheets
        excel_data = pd.ExcelFile(excel_file)
        sheets = excel_data.sheet_names
        
        print(f"📊 Number of sheets: {len(sheets)}")
        print(f"📋 Sheet names: {sheets}")
        print()
        
        results = {}
        
        # Analyze each sheet
        for sheet_name in sheets:
            print(f"🔸 Analyzing sheet: '{sheet_name}'")
            print("-" * 40)
            
            try:
                # Load the sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Basic info
                print(f"   📏 Shape: {df.shape} (rows x columns)")
                print(f"   🏷️  Columns: {list(df.columns)}")
                
                # Check for date column
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_columns:
                    print(f"   📅 Date columns found: {date_columns}")
                    
                    # Analyze date range
                    for date_col in date_columns:
                        try:
                            df[date_col] = pd.to_datetime(df[date_col])
                            min_date = df[date_col].min()
                            max_date = df[date_col].max()
                            print(f"   📈 Date range ({date_col}): {min_date.date()} to {max_date.date()}")
                            print(f"   📊 Total days: {(max_date - min_date).days}")
                        except:
                            print(f"   ⚠️  Could not parse dates in column: {date_col}")
                
                # Look for OHLC columns (including Price as Close)
                ohlc_columns = []
                price_mapping = {'price': 'close', 'vol.': 'volume'}
                
                for col in df.columns:
                    col_lower = col.lower()
                    if any(price_type in col_lower for price_type in ['open', 'high', 'low', 'close', 'volume', 'price', 'vol.']):
                        ohlc_columns.append(col)
                
                if ohlc_columns:
                    print(f"   💰 OHLC columns found: {ohlc_columns}")
                    
                    # Check data quality
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        print(f"   🔢 Numeric columns: {len(numeric_cols)}")
                        
                        # Sample data ranges
                        for col in numeric_cols[:5]:  # Show first 5 numeric columns
                            if not df[col].empty:
                                min_val = df[col].min()
                                max_val = df[col].max()
                                print(f"      {col}: {min_val:.2f} - {max_val:.2f}")
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Check for missing data
                missing_data = df.isnull().sum()
                total_missing = missing_data.sum()
                if total_missing > 0:
                    print(f"   ⚠️  Missing values: {total_missing} total")
                    missing_cols = missing_data[missing_data > 0]
                    for col, count in missing_cols.items():
                        print(f"      {col}: {count} missing")
                else:
                    print(f"   ✅ No missing values")
                
                # Sample of data
                print(f"   📋 First 3 rows:")
                print(df.head(3).to_string(max_cols=6))
                
                # Store results
                results[sheet_name] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'date_columns': date_columns,
                    'ohlc_columns': ohlc_columns,
                    'numeric_columns': numeric_cols,
                    'missing_values': total_missing,
                    'sample_data': df.head(3)
                }
                
            except Exception as e:
                print(f"   ❌ Error analyzing sheet '{sheet_name}': {e}")
                results[sheet_name] = {'error': str(e)}
            
            print()
        
        # Summary analysis
        print("📊 SUMMARY ANALYSIS")
        print("=" * 60)
        
        # Determine instrument organization
        instrument_pattern = None
        if len(sheets) == 3:
            print("🎯 Detected: Likely 3 separate sheets for 3 instruments")
            instrument_pattern = "separate_sheets"
        elif len(sheets) == 1:
            print("🎯 Detected: Likely single sheet with multiple instruments")
            instrument_pattern = "single_sheet"
        
        # Check for standard OHLC format
        standard_ohlc = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        compatible_sheets = []
        
        for sheet_name, data in results.items():
            if 'error' not in data:
                sheet_cols = [col.lower() for col in data['columns']]
                # Check for OHLC including Price as Close equivalent
                has_open = 'open' in sheet_cols
                has_high = 'high' in sheet_cols  
                has_low = 'low' in sheet_cols
                has_close = 'close' in sheet_cols or 'price' in sheet_cols
                has_date = any('date' in col.lower() for col in data['columns'])
                
                if has_open and has_high and has_low and has_close and has_date:
                    compatible_sheets.append(sheet_name)
                    print(f"✅ Sheet '{sheet_name}' is OHLC compatible")
                else:
                    missing = []
                    if not has_open: missing.append('Open')
                    if not has_high: missing.append('High') 
                    if not has_low: missing.append('Low')
                    if not has_close: missing.append('Close/Price')
                    if not has_date: missing.append('Date')
                    print(f"⚠️  Sheet '{sheet_name}' missing: {missing}")
        
        print(f"\n🎯 Compatible sheets for training: {len(compatible_sheets)}")
        print(f"📋 Sheet names: {compatible_sheets}")
        
        # Instrument mapping suggestions
        print(f"\n🏷️  SUGGESTED INSTRUMENT MAPPING:")
        if len(compatible_sheets) >= 3:
            print(f"   NASDAQ: {compatible_sheets[0] if len(compatible_sheets) > 0 else 'TBD'}")
            print(f"   SP500:  {compatible_sheets[1] if len(compatible_sheets) > 1 else 'TBD'}")
            print(f"   DOW:    {compatible_sheets[2] if len(compatible_sheets) > 2 else 'TBD'}")
        else:
            print(f"   Available sheets: {compatible_sheets}")
        
        # Expected data volume
        total_rows = sum(data.get('shape', [0, 0])[0] for data in results.values() if 'error' not in data)
        print(f"\n📈 EXPECTED DATASET SIZE:")
        print(f"   Total rows across sheets: {total_rows:,}")
        print(f"   Expected training samples: ~{total_rows:,} (with 30-bar charts)")
        
        print(f"\n✅ Excel structure analysis completed!")
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing Excel file: {e}")
        return None

def suggest_mapping_config(analysis_results):
    """Suggest configuration mapping based on analysis."""
    if not analysis_results:
        return None
    
    print(f"\n🔧 SUGGESTED CONFIGURATION:")
    print("=" * 60)
    
    # Determine sheets to use
    compatible_sheets = []
    for sheet_name, data in analysis_results.items():
        if 'error' not in data and 'ohlc_columns' in data and data['ohlc_columns']:
            compatible_sheets.append(sheet_name)
    
    if len(compatible_sheets) >= 1:
        print("data:")
        print("  source: 'excel'")
        print("  excel_file: 'data/Futures_Data_Consolidated.xlsx'")
        print("  instruments:")
        
        # Map sheets to instruments
        instrument_names = ['NASDAQ', 'SP500', 'DOW']
        for i, sheet in enumerate(compatible_sheets[:3]):
            instrument = instrument_names[i] if i < len(instrument_names) else f'INSTRUMENT_{i+1}'
            print(f"    {instrument}:")
            print(f"      sheet: '{sheet}'")
            
            # Suggest column mapping
            if sheet in analysis_results:
                columns = analysis_results[sheet].get('columns', [])
                print(f"      columns:")
                
                # Find date column
                date_col = next((col for col in columns if 'date' in col.lower()), 'Date')
                print(f"        date: '{date_col}'")
                
                # Find OHLC columns with proper mapping
                col_mapping = {
                    'open': next((c for c in columns if 'open' in c.lower()), 'Open'),
                    'high': next((c for c in columns if 'high' in c.lower()), 'High'), 
                    'low': next((c for c in columns if 'low' in c.lower()), 'Low'),
                    'close': next((c for c in columns if 'price' in c.lower() or 'close' in c.lower()), 'Price'),
                    'volume': next((c for c in columns if 'vol' in c.lower()), 'Vol.')
                }
                
                for price_type, col_name in col_mapping.items():
                    print(f"        {price_type}: '{col_name}'")
        
        print(f"\n  available_instruments: {instrument_names[:len(compatible_sheets)]}")
    
    return compatible_sheets

if __name__ == "__main__":
    print("🚀 Excel Data Structure Analysis")
    print("=" * 60)
    
    # Analyze the Excel file
    results = analyze_excel_structure()
    
    if results:
        # Suggest configuration
        suggest_mapping_config(results)
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Create ExcelDataFetcher based on this analysis")
        print(f"   2. Update config with suggested instrument mapping")
        print(f"   3. Test data loading with different instrument selections")
    else:
        print(f"\n❌ Analysis failed. Please check the Excel file.")