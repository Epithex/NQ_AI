# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: When making changes that affect project context, architecture, or workflow, update this CLAUDE.md file to reflect the new state.

## Project Overview

NQ_AI is a pure visual AI trading system that analyzes daily candlestick patterns across major US index futures using computer vision. The system uses ViT-Base (87M parameters) to classify daily chart patterns based solely on visual information for bullish/bearish sentiment analysis.

**Core Function**: Binary visual classification system that predicts daily candle sentiment:
- **0**: Bearish (Close < Open - red candle)
- **1**: Bullish (Close > Open - green candle)  
- **2**: Neutral (Close = Open - doji candle, very rare)

**Multi-Instrument Approach**: Trains on 3 highly correlated US index futures for robust pattern learning:
- **DOW**: Dow Jones E-mini futures (DowJones Excel sheet)
- **NASDAQ**: NASDAQ-100 E-mini futures (Nasdaq Excel sheet)  
- **SP500**: S&P 500 E-mini futures (SP500 Excel sheet)
- **Combined Dataset**: ~19,000 samples across all instruments (2000-2025) from Excel data source

## Development Commands

```bash
# Virtual environment (Python 3.13)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Interactive development (Cursor IDE with Python Interactive Window for cell-by-cell execution)
jupyter lab

# Code formatting (PEP 8 compliant)
black .

# Linting
ruff check .
```

## Python Development Standards

### Code Quality Requirements
- **PEP 8 Compliance**: All code must follow Python style guide
- **Type Hints**: Required for all function signatures (e.g., `def func(data: pd.DataFrame) -> dict:`)
- **Docstrings**: Google-style docstrings for all classes and methods
- **Auto-formatting**: Use Black for consistent code formatting
- **Linting**: Use Ruff for code quality checks

### Best Practices
- **Configuration Management**: Use `config.yaml` for all settings (paths, tickers, date ranges)
- **Modularity**: Encapsulate logic in defined classes and methods
- **No Hard-coding**: All parameters must be configurable
- **Interactive Development**: Use Cursor IDE with Python Interactive Window feature (powered by Jupyter extension) for cell-by-cell execution within .py scripts
- **Central Script**: Main implementation in single script developed interactively
- **Version Control**: Git repository with `.gitignore` for venv and large data files

## System Architecture

The system implements a pure visual pattern analysis pipeline with four core components:

### 1. Excel Data Fetcher
- Loads daily OHLC data directly from `Futures_Data_Consolidated.xlsx` file
- Handles multiple instruments with sheet-based organization (DOW→DowJones, NASDAQ→Nasdaq, SP500→SP500)
- Supports flexible instrument selection (single, multiple, or ALL instruments)
- Key methods:
  - `fetch_instrument_data(instrument, start_year, end_year)`: Load data for specific instrument from Excel
  - `fetch_multi_instrument_data(instruments, start_year, end_year)`: Load data for multiple instruments
  - `validate_data_quality(data, instrument)`: Ensure data integrity and completeness
  - `clean_excel_data(data, instrument)`: Standardize Excel data format (Price→Close mapping)

### 2. Pure Visual Chart Generator
- Creates clean training images showing 30 daily bars for binary sentiment analysis
- Generates charts optimized for pure visual pattern recognition (224x224 pixels)
- **Binary Classification Focus**: No previous day reference lines - pure candlestick patterns only
- Key methods:
  - `generate_binary_chart_image(date, data, instrument)`: Create clean chart for sentiment analysis
  - `save_chart_image(image, filename)`: Save chart image for training dataset
  - **No numerical overlays**: Pure visual information for open vs close analysis

### 3. Binary Pattern Analyzer
- Analyzes single trading day to determine binary sentiment classification
- Compares daily open vs close prices for bullish/bearish determination
- Applies consistent binary logic across all instruments (DOW, NASDAQ, SP500)
- Key methods:
  - `analyze_binary_pattern(daily_candle)`: Determine binary sentiment (0=Bearish, 1=Bullish, 2=Neutral)
  - `extract_candle_features(daily_candle)`: Extract metadata for analysis
  - `get_pattern_statistics(patterns)`: Calculate distribution and balance metrics

### 4. Multi-Instrument Dataset Creator
- Orchestrates the complete Excel-based dataset generation workflow
- Creates training samples with pure visual charts and binary sentiment labels
- Supports flexible instrument selection (single, multiple, or ALL) with date filtering
- Key methods:
  - `create_multi_instrument_dataset(instruments, start_year, end_year)`: Main workflow orchestrator
  - `create_binary_samples(instrument, data)`: Generate samples with charts + binary labels
  - `resolve_instruments(selected_instruments)`: Handle instrument selection including 'ALL' option
  - `save_complete_dataset()`: Export dataset manifest and sample indices for training

## Key Technical Requirements

### Data Sources
- **Excel File**: `data/Futures_Data_Consolidated.xlsx` with 3 instrument sheets
  - **DowJones Sheet**: Dow Jones futures data (6,076 bars: 2002-2025)
  - **Nasdaq Sheet**: NASDAQ-100 futures data (6,584 bars: 2000-2025)  
  - **SP500 Sheet**: S&P 500 futures data (6,564 bars: 2000-2025)
- **Column Mapping**: Date, Price→Close, Open, High, Low, Vol.→Volume
- **mplfinance**: Chart generation for pure visual model input
- **openpyxl**: Excel file reading and processing

### Binary Visual Analysis Framework
- **Chart Timeframe**: Daily bars
- **Chart Duration**: 30 daily bars (approximately 6 weeks context) for each training image
- **Pattern Analysis**: Compare daily open vs close prices (visual only)
- **Classification Method**: Simple open vs close comparison for bullish/bearish sentiment
- **Visual Elements**: Clean candlestick charts with no reference lines or overlays
- **No Numerical Features**: Pure visual learning, no numerical overlays or annotations

### Multi-Instrument Data Splits (Strict Separation)
- **Training Set**: 2000-2020 (20 years historical data across all instruments)
- **Validation Set**: 2021-2022 (2 years model tuning across all instruments)
- **Test Set**: 2023-2025 (3 years final evaluation across all instruments)
- **Total Dataset**: ~20,000 samples spanning 25 years (2000-2025) across 3 instruments
- **No Data Leakage**: Strict temporal separation prevents look-ahead bias

### Binary Visual Classification Logic
1. **Chart Generation**: Create clean 30-bar daily chart without reference lines or overlays
2. **Pure Visual Input**: No numerical features - chart contains all necessary information visually
3. **Binary Classification** (3-class system):
   - **Label 0**: Bearish (close < open - red candle)
   - **Label 1**: Bullish (close > open - green candle)
   - **Label 2**: Neutral (close = open - doji candle, very rare)

4. **Training Sample**:
   - **Image**: Clean 30-bar daily chart (224x224 pixels)
   - **Label**: Binary sentiment classification (0-2)
   - **Instrument**: Instrument identifier (DOW, NASDAQ, or SP500)
   - **No Numerical Features**: Pure visual learning approach

**Example**: Daily open at 15,100, close at 15,150. Since close > open, this is a bullish candle = Label 1 (Bullish).

## Project Structure

```
/src
  # Binary Classification System (Current)
  binary_pattern_analyzer.py   # Binary bullish/bearish classification logic
  binary_chart_generator.py    # Clean chart generation (no reference lines)
  binary_dataset_creator.py    # Multi-instrument binary dataset creation
  binary_data_loader.py        # TensorFlow data pipeline for binary training
  binary_vit_model.py          # Binary ViT-Base (87M params) for sentiment
  train_binary_vit.py          # Binary classification training pipeline
  
  # Legacy 4-class system (for reference)
  daily_pattern_analyzer.py    # 4-class pattern logic - LEGACY
  daily_chart_generator.py     # Charts with prev day lines - LEGACY
  pure_visual_vit_model.py     # 4-class ViT-Base - LEGACY
  
  # Excel-based data fetching
  excel_data_fetcher.py        # Excel-based multi-instrument data fetching
  
/data
  Futures_Data_Consolidated.xlsx  # Excel data source (DowJones, Nasdaq, SP500 sheets)
  /images_binary               # Binary classification chart images (224x224)
  /labels_binary               # Binary sentiment labels and metadata
  /metadata_binary             # Binary dataset manifests and indices
  
/models
  /outputs_binary              # Binary model training artifacts and results
  
/config
  config_binary_visual.yaml    # Binary ViT-Base configuration (Current)
  # Legacy configurations (for reference)
  config_pure_visual.yaml      # 4-class ViT-Base - LEGACY
  config.yaml                  # Custom hybrid ViT - LEGACY

# Main Scripts
generate_binary_dataset.py    # Generate complete binary dataset
```

## Model Architecture

### Binary Visual ViT-Base (87M parameters)  
- Google ViT-Base-Patch16-224 architecture
- 12 transformer layers, 768 hidden size, 12 attention heads
- **Pure visual input**: 224x224 chart images only
- **Binary sentiment output**: 3-class classification (Bearish/Bullish/Neutral)
- **No numerical features**: Eliminates fusion complexity
- **Multi-instrument training**: Learns patterns across DOW, NASDAQ, SP500
- Optimized for bullish/bearish sentiment analysis on candlestick charts

## Development Priorities

1. **Excel Data Integration**: Direct Excel file processing ✅ COMPLETE
2. **Binary Dataset Generation**: ~19,000 samples across DOW, NASDAQ, SP500 (2000-2025) ✅ READY
3. **Binary ViT-Base Model**: 87M parameter binary sentiment model ✅ READY
4. **Training Pipeline**: Streamlined binary classification training ✅ READY
5. **RunPod Deployment**: GPU training on complete dataset ⏳ PENDING

## Important Implementation Notes

- **Chart Display**: Show 30 daily bars (approximately 6 weeks) of price action
- **Clean Visualization**: No reference lines, overlays, or numerical annotations
- **Sentiment Analysis**: Compare daily open vs close prices for bullish/bearish classification
- **Pure Visual Input**: Only chart images (224x224), no numerical features
- **Multi-Instrument**: Same sentiment logic applied across DOW, NASDAQ, SP500 futures
- **No Data Augmentation**: Preserve chart integrity for financial data
- **Simple Logic**: Focus only on open vs close comparison for sentiment classification

## Current Status

✅ **Excel Data Integration**: Complete Excel-based pipeline replacing API dependencies
✅ **Binary Classification System**: Complete implementation ready for training
✅ **Legacy Systems**: 4-class and hybrid models available for reference
✅ **Binary ViT-Base Model**: 87M parameter architecture optimized for sentiment
✅ **Multi-Instrument Pipeline**: DOW + NASDAQ + SP500 dataset generation ready
✅ **Training Pipeline**: Comprehensive training system ready for RunPod deployment

## Performance Metrics (Tested)

- **NASDAQ 2024**: 226 training samples from 255 raw bars (88.6% efficiency)
- **Pattern Distribution**: 44.2% Bearish, 55.8% Bullish, 0% Neutral
- **Chart Generation**: ~0.1 seconds per chart image (224x224 PNG)
- **Date Filtering**: Proper start_year/end_year parameter handling
- **No API Calls**: Direct Excel processing with openpyxl

### Parallel Processing Performance
- **Single-threaded**: ~3-5 samples/second (original)
- **8 workers**: ~20-30 samples/second (6-8x speedup)
- **16 workers**: ~40-50 samples/second (10-15x speedup)
- **32 workers**: ~60-80 samples/second (15-25x speedup)
- **Full Dataset (ALL instruments)**: ~2-5 minutes vs ~1-2 hours (sequential)
- **Memory Usage**: ~500MB per worker process
- **Resource Management**: Auto-cleanup, interrupt handling, progress tracking

## Usage

### Excel-Based Dataset Generation

```bash
# Generate binary classification dataset for all instruments (DOW, NASDAQ, SP500)
python generate_binary_dataset.py --instruments ALL

# Generate dataset for single instrument
python generate_binary_dataset.py --instruments NASDAQ

# Generate dataset for multiple specific instruments  
python generate_binary_dataset.py --instruments DOW,SP500

# Generate with custom date range
python generate_binary_dataset.py --instruments NASDAQ --start_year 2020 --end_year 2023

# Dry run to validate configuration and instrument selection
python generate_binary_dataset.py --instruments ALL --dry_run

# Small test dataset (single year)
python generate_binary_dataset.py --instruments NASDAQ --start_year 2024 --end_year 2024

# PARALLEL PROCESSING (NEW)
# Enable parallel processing with auto-detected workers
python generate_binary_dataset.py --instruments ALL --parallel

# Specify number of parallel workers (max 32)
python generate_binary_dataset.py --instruments ALL --workers 16

# Maximum parallel performance
python generate_binary_dataset.py --instruments ALL --workers 32

# Parallel with custom date range
python generate_binary_dataset.py --instruments ALL --workers 16 --start_year 2020 --end_year 2024
```

### Model Training

```bash
# Train binary ViT-Base model (after dataset generation)
python src/train_binary_vit.py --config config/config_binary_visual.yaml

# Verify dataset integrity
python generate_binary_dataset.py verify

# Clean up old dataset files
python generate_binary_dataset.py cleanup
```

### Available Instruments

- **DOW**: Dow Jones E-mini futures (2002-2025, ~6,000 bars)
- **NASDAQ**: NASDAQ-100 E-mini futures (2000-2025, ~6,500 bars)  
- **SP500**: S&P 500 E-mini futures (2000-2025, ~6,500 bars)
- **ALL**: Process all three instruments together

### Command Line Options

```
--instruments INSTRUMENTS    Comma-separated list (DOW, NASDAQ, SP500, ALL)
--start_year START_YEAR      Override start year (default: 2000)
--end_year END_YEAR          Override end year (default: 2025) 
--dry_run                    Validate configuration without generating data
--config CONFIG              Custom configuration file path
--parallel                   Enable parallel processing with auto-detected workers
--workers WORKERS            Number of parallel workers (max 32, default: single-threaded)
```

## Excel Data Structure

### File Organization
- **File**: `data/Futures_Data_Consolidated.xlsx`
- **Sheets**: 4 total (Summary + 3 instrument sheets)
  - `DowJones`: 6,076 bars (2002-04-05 to 2025-08-15)
  - `Nasdaq`: 6,584 bars (2000-01-18 to 2025-08-15)  
  - `SP500`: 6,564 bars (2000-01-18 to 2025-08-15)

### Column Mapping
- **Date**: Trading date (index)
- **Price**: Close price (mapped to Close)
- **Open**: Opening price
- **High**: Daily high price  
- **Low**: Daily low price
- **Vol.**: Volume (mapped to Volume)
- **Change %**: Daily percentage change (metadata only)

### Data Quality Features
- **OHLC Validation**: Ensures High ≥ Open,Close,Low and Low ≤ Open,Close
- **Missing Data Handling**: Fills missing volume with 0, removes invalid OHLC bars
- **Date Filtering**: Supports custom start_year/end_year ranges
- **Weekend Removal**: Filters out non-trading days automatically

## Migration from API to Excel

### What Changed
- ❌ **Removed**: yfinance, requests, multitasking dependencies
- ❌ **Removed**: stooq_fetcher.py API-based data fetching
- ✅ **Added**: excel_data_fetcher.py for direct Excel processing
- ✅ **Added**: openpyxl dependency for Excel file reading
- ✅ **Enhanced**: Flexible instrument selection (single, multiple, ALL)
- ✅ **Enhanced**: Better date range filtering and validation

### Benefits of Excel Approach
- **Reliability**: No API failures, rate limits, or network dependencies
- **Speed**: Faster data access (local file vs network calls)
- **Consistency**: Same data every run, no API data variations
- **Flexibility**: Easy instrument selection without configuration changes
- **Deployment**: Simplified deployment without API keys or network access

