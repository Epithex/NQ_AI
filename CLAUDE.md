# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: When making changes that affect project context, architecture, or workflow, update this CLAUDE.md file to reflect the new state.

## Project Overview

NQ_AI is a hybrid AI trading system that analyzes daily candlestick patterns across major US index futures using a combination of computer vision and numerical features. The system uses a hybrid ViT-Base architecture (87M+ parameters) to classify daily chart patterns based on previous day levels interactions with both visual and numerical feature fusion.

**Core Function**: 4-class hybrid classification system that predicts daily pattern interactions with previous day levels:
- **1**: High Breakout (current_high >= prev_high && current_low > prev_low)
- **2**: Low Breakdown (current_low <= prev_low && current_high < prev_high)  
- **3**: Range Expansion (current_high >= prev_high && current_low <= prev_low)
- **4**: Range Bound (current_high < prev_high && current_low > prev_low)

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
- **Configuration Management**: Use `config_daily_hybrid.yaml` for all settings (paths, tickers, date ranges)
- **Modularity**: Encapsulate logic in defined classes and methods
- **No Hard-coding**: All parameters must be configurable
- **Interactive Development**: Use Cursor IDE with Python Interactive Window feature (powered by Jupyter extension) for cell-by-cell execution within .py scripts
- **Central Script**: Main implementation in single script developed interactively
- **Version Control**: Git repository with `.gitignore` for venv and large data files

## System Architecture

The system implements a hybrid pattern analysis pipeline with four core components:

### 1. Excel Data Fetcher
- Loads daily OHLC data directly from `Futures_Data_Consolidated.xlsx` file
- Handles multiple instruments with sheet-based organization (DOW→DowJones, NASDAQ→Nasdaq, SP500→SP500)
- Supports flexible instrument selection (single, multiple, or ALL instruments)
- Key methods:
  - `fetch_instrument_data(instrument, start_year, end_year)`: Load data for specific instrument from Excel
  - `fetch_multi_instrument_data(instruments, start_year, end_year)`: Load data for multiple instruments
  - `validate_data_quality(data, instrument)`: Ensure data integrity and completeness
  - `clean_excel_data(data, instrument)`: Standardize Excel data format (Price→Close mapping)

### 2. Daily Hybrid Chart Generator
- Creates training images showing 30 daily bars with previous day level reference lines
- Generates charts optimized for hybrid pattern recognition (224x224 pixels)
- **Previous Day Levels Focus**: Green line for previous day high, red line for previous day low
- Key methods:
  - `generate_daily_chart_image(date, data, previous_candle, instrument)`: Create chart with reference lines
  - `save_chart_image(image, filename)`: Save chart image for training dataset
  - **Visual + Reference Lines**: Charts include previous day level lines for pattern analysis

### 3. Daily Pattern Analyzer
- Analyzes daily trading patterns relative to previous day high/low levels
- Compares current day high/low vs previous day high/low for 4-class classification
- Applies consistent logic across all instruments (DOW, NASDAQ, SP500)
- Key methods:
  - `analyze_daily_pattern(current_candle, previous_candle)`: Determine 4-class pattern (1-4)
  - `extract_candle_features(current_candle, previous_candle)`: Extract metadata for analysis
  - `extract_numerical_features(current_candle, previous_candle)`: Extract 3 key numerical features
  - `get_pattern_statistics(patterns)`: Calculate distribution and balance metrics

### 4. Multi-Instrument Hybrid Dataset Creator
- Orchestrates the complete Excel-based hybrid dataset generation workflow
- Creates training samples with visual charts, previous day reference lines, and numerical features
- Supports flexible instrument selection (single, multiple, or ALL) with date filtering and parallel processing
- Key methods:
  - `create_multi_instrument_dataset(instruments, start_year, end_year)`: Main workflow orchestrator
  - `create_daily_samples(instrument, data)`: Generate hybrid samples with charts + numerical features
  - `create_daily_samples_parallel(instrument, data)`: Parallel processing for large datasets
  - `resolve_instruments(selected_instruments)`: Handle instrument selection including 'ALL' option
  - `save_dataset_manifest()`: Export dataset manifest and sample indices for training

## Key Technical Requirements

### Data Sources
- **Excel File**: `data/Futures_Data_Consolidated.xlsx` with 3 instrument sheets
  - **DowJones Sheet**: Dow Jones futures data (6,076 bars: 2002-2025)
  - **Nasdaq Sheet**: NASDAQ-100 futures data (6,584 bars: 2000-2025)  
  - **SP500 Sheet**: S&P 500 futures data (6,564 bars: 2000-2025)
- **Column Mapping**: Date, Price→Close, Open, High, Low, Vol.→Volume
- **mplfinance**: Chart generation with previous day level reference lines
- **openpyxl**: Excel file reading and processing

### Hybrid Visual + Numerical Analysis Framework
- **Chart Timeframe**: Daily bars
- **Chart Duration**: 30 daily bars (approximately 6 weeks context) for each training image
- **Pattern Analysis**: Compare current high/low vs previous day high/low levels
- **Classification Method**: 4-class previous day levels interaction analysis
- **Visual Elements**: Candlestick charts with green (prev high) and red (prev low) reference lines
- **Numerical Features**: 3 key features (distance_to_prev_high, distance_to_prev_low, prev_day_range)

### Multi-Instrument Data Splits (Strict Separation)
- **Training Set**: 2000-2020 (20 years historical data across all instruments)
- **Validation Set**: 2021-2022 (2 years model tuning across all instruments)
- **Test Set**: 2023-2025 (3 years final evaluation across all instruments)
- **Total Dataset**: ~19,000 samples spanning 25 years (2000-2025) across 3 instruments
- **No Data Leakage**: Strict temporal separation prevents look-ahead bias

### 4-Class Hybrid Classification Logic
1. **Chart Generation**: Create 30-bar daily chart with previous day high/low reference lines
2. **Numerical Feature Extraction**: Calculate 3 key metrics for current vs previous day
3. **4-Class Pattern Analysis**:
   - **Pattern 1**: High Breakout (current_high >= prev_high && current_low > prev_low)
   - **Pattern 2**: Low Breakdown (current_low <= prev_low && current_high < prev_high)
   - **Pattern 3**: Range Expansion (current_high >= prev_high && current_low <= prev_low)
   - **Pattern 4**: Range Bound (current_high < prev_high && current_low > prev_low)

4. **Training Sample**:
   - **Image**: 30-bar daily chart with reference lines (224x224 pixels)
   - **Numerical Features**: 3-element array [distance_to_prev_high, distance_to_prev_low, prev_day_range]
   - **Label**: 4-class pattern classification (1-4, mapped to 0-3 for TensorFlow)
   - **Instrument**: Instrument identifier (DOW, NASDAQ, or SP500)

**Example**: Current high=15,200, low=15,050. Previous high=15,100, low=15,000. Since current_high >= prev_high (15,200 >= 15,100) AND current_low > prev_low (15,050 > 15,000), this is Pattern 1 (High Breakout).

## Project Structure

```
/src
  # 4-Class Hybrid System (Current)
  daily_pattern_analyzer.py    # 4-class previous day levels classification logic
  daily_chart_generator.py     # Chart generation with previous day level reference lines
  daily_dataset_creator.py     # Multi-instrument hybrid dataset creation with parallel processing
  daily_data_loader.py         # TensorFlow data pipeline for hybrid training (dual inputs)
  hybrid_vit_model.py          # Hybrid ViT-Base (87M+ params) with visual + numerical fusion
  train_daily_model.py         # 4-class hybrid classification training pipeline
  
  # Legacy binary system (for reference)
  binary_pattern_analyzer.py   # Binary bullish/bearish logic - LEGACY
  binary_chart_generator.py    # Clean charts (no reference lines) - LEGACY
  binary_dataset_creator.py    # Binary dataset creation - LEGACY
  binary_data_loader.py        # Binary data pipeline - LEGACY
  binary_vit_model.py          # Binary ViT-Base - LEGACY
  train_binary_vit.py          # Binary training pipeline - LEGACY
  
  # Excel-based data fetching
  excel_data_fetcher.py        # Excel-based multi-instrument data fetching
  
/data
  Futures_Data_Consolidated.xlsx  # Excel data source (DowJones, Nasdaq, SP500 sheets)
  /images_daily                # 4-class hybrid chart images (224x224) with reference lines
  /labels_daily                # 4-class hybrid labels and metadata
  /metadata_daily              # Hybrid dataset manifests and indices
  
/models
  /outputs_daily               # Hybrid model training artifacts and results
  
/config
  config_daily_hybrid.yaml     # 4-class hybrid ViT configuration (Current)
  # Legacy configurations (for reference)
  config_binary_visual.yaml    # Binary ViT-Base - LEGACY
  config_pure_visual.yaml      # 4-class ViT-Base - LEGACY
  config.yaml                  # Custom hybrid ViT - LEGACY

# Main Scripts
generate_daily_dataset.py     # Generate complete 4-class hybrid dataset
```

## Model Architecture

### Hybrid Daily ViT-Base (87M+ parameters)  
- Google ViT-Base-Patch16-224 architecture with feature fusion
- 12 transformer layers, 768 hidden size, 12 attention heads (visual branch)
- **Dual input architecture**: 224x224 chart images + 3 numerical features
- **4-class pattern output**: Previous day levels classification (High Breakout/Low Breakdown/Range Expansion/Range Bound)
- **Feature fusion**: Early fusion combining visual ViT features with numerical features
- **Multi-instrument training**: Learns patterns across DOW, NASDAQ, SP500
- Optimized for previous day levels interaction analysis

#### Architecture Details
- **Visual Branch**: ViT-Base processes 224x224 chart images with reference lines
- **Numerical Branch**: Dense layers [64, 32, 16] process 3 key features
- **Feature Fusion**: Concatenation + dense layers [512, 256] for combined processing
- **Classification Head**: 4-class output with softmax activation
- **Total Parameters**: 87M+ (ViT-Base + fusion layers)

## Development Priorities

1. **4-Class Hybrid Implementation**: Complete hybrid system with visual + numerical fusion ✅ COMPLETE
2. **Parallel Processing**: Multi-worker dataset generation with ProcessPoolExecutor ✅ COMPLETE
3. **Hybrid Training Pipeline**: TensorFlow training with dual inputs ✅ COMPLETE
4. **Previous Day Levels Strategy**: 4-class pattern classification system ✅ COMPLETE
5. **RunPod Deployment**: GPU training on complete hybrid dataset ⏳ PENDING

## Important Implementation Notes

- **Chart Display**: Show 30 daily bars (approximately 6 weeks) with previous day level reference lines
- **Reference Lines**: Green line for previous day high, red line for previous day low
- **Pattern Analysis**: Compare current day high/low vs previous day high/low levels
- **Hybrid Input**: Both chart images (224x224) and numerical features (3 elements)
- **Multi-Instrument**: Same pattern logic applied across DOW, NASDAQ, SP500 futures
- **No Data Augmentation**: Preserve chart integrity for financial data
- **Complex Logic**: 4-class pattern system based on level interactions

## Current Status

✅ **4-Class Hybrid System**: Complete implementation with visual + numerical feature fusion  
✅ **Previous Day Levels Strategy**: 4-class pattern classification fully implemented  
✅ **Parallel Processing**: Multi-worker dataset generation with ProcessPoolExecutor  
✅ **Hybrid ViT Model**: 87M+ parameter architecture with dual inputs  
✅ **Training Pipeline**: Complete hybrid training system with TensorFlow  
✅ **Excel Data Integration**: Complete Excel-based pipeline  
✅ **Multi-Instrument Pipeline**: DOW + NASDAQ + SP500 hybrid dataset generation ready  
✅ **Configuration System**: Complete config_daily_hybrid.yaml setup  

## Performance Metrics (Projected)

- **Expected Dataset Size**: ~19,000 hybrid samples across all instruments
- **4-Class Pattern Distribution**: ~25% each pattern (High Breakout, Low Breakdown, Range Expansion, Range Bound)
- **Chart Generation**: ~0.1 seconds per chart image with reference lines (224x224 PNG)
- **Numerical Feature Extraction**: ~0.01 seconds per sample (3 features)
- **No API Calls**: Direct Excel processing with openpyxl

### Parallel Processing Performance
- **Single-threaded**: ~3-5 samples/second (original)
- **8 workers**: ~20-30 samples/second (6-8x speedup)
- **16 workers**: ~40-50 samples/second (10-15x speedup)
- **32 workers**: ~60-80 samples/second (15-25x speedup)
- **Full Hybrid Dataset (ALL instruments)**: ~3-6 minutes vs ~2-3 hours (sequential)
- **Memory Usage**: ~1GB per worker process (increased for hybrid processing)
- **Resource Management**: Auto-cleanup, interrupt handling, progress tracking

## Usage

### Excel-Based Hybrid Dataset Generation

```bash
# Generate 4-class hybrid dataset for all instruments (DOW, NASDAQ, SP500)
python generate_daily_dataset.py --instruments ALL

# Generate dataset for single instrument
python generate_daily_dataset.py --instruments NASDAQ

# Generate dataset for multiple specific instruments  
python generate_daily_dataset.py --instruments DOW,SP500

# Generate with custom date range
python generate_daily_dataset.py --instruments NASDAQ --start_year 2020 --end_year 2023

# Dry run to validate configuration and instrument selection
python generate_daily_dataset.py --instruments ALL --dry_run

# Small test dataset (single year)
python generate_daily_dataset.py --instruments NASDAQ --start_year 2024 --end_year 2024

# PARALLEL PROCESSING (RECOMMENDED)
# Enable parallel processing with auto-detected workers
python generate_daily_dataset.py --instruments ALL --parallel

# Specify number of parallel workers (max 32)
python generate_daily_dataset.py --instruments ALL --workers 16

# Maximum parallel performance
python generate_daily_dataset.py --instruments ALL --workers 32

# Parallel with custom date range
python generate_daily_dataset.py --instruments ALL --workers 16 --start_year 2020 --end_year 2024
```

### Hybrid Model Training

```bash
# Train 4-class hybrid ViT model (after dataset generation)
python src/train_daily_model.py --config config/config_daily_hybrid.yaml

# Verify hybrid dataset integrity
python generate_daily_dataset.py verify

# Clean up old dataset files
python generate_daily_dataset.py cleanup

# Show detailed help
python generate_daily_dataset.py help
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

## Migration from Binary to 4-Class Hybrid

### What Changed
- ❌ **Removed**: Binary 3-class sentiment analysis (Bearish/Bullish/Neutral)
- ❌ **Removed**: Pure visual approach without numerical features
- ✅ **Added**: 4-class previous day levels pattern analysis
- ✅ **Added**: Hybrid ViT model with visual + numerical feature fusion
- ✅ **Added**: Previous day high/low reference lines in charts
- ✅ **Added**: 3 numerical features for enhanced pattern recognition
- ✅ **Enhanced**: More sophisticated pattern classification system
- ✅ **Enhanced**: Parallel processing for hybrid dataset generation

### Benefits of Hybrid Approach
- **Enhanced Accuracy**: Combines visual patterns with numerical metrics
- **Sophisticated Analysis**: 4-class system captures more nuanced market behavior
- **Previous Day Context**: Reference lines provide crucial level interaction information
- **Feature Fusion**: Leverages both computer vision and numerical analysis
- **Scalable Processing**: Parallel processing handles increased computational requirements