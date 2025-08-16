# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: When making changes that affect project context, architecture, or workflow, update this CLAUDE.md file to reflect the new state.

## Project Overview

NQ_AI is a pure visual AI trading system that analyzes daily price patterns across major US index futures using computer vision. The system uses ViT-Base (87M parameters) to classify daily chart patterns based solely on visual information, eliminating the complexity of hybrid numerical-visual approaches.

**Core Function**: Four-class visual classification system that predicts daily patterns:
- **1**: High Breakout (Daily high >= previous day high only)
- **2**: Low Breakdown (Daily low <= previous day low only)  
- **3**: Range Expansion (Both levels touched during the day)
- **4**: Range Bound (Neither level touched during the day)

**Multi-Instrument Approach**: Trains on 3 highly correlated US index futures for robust pattern learning:
- **NQ.F**: NASDAQ-100 E-mini futures (primary instrument)
- **ES.F**: S&P 500 E-mini futures (high correlation to NQ ~0.9)
- **YM.F**: Dow Jones E-mini futures (additional pattern diversity)
- **Combined Dataset**: ~20,000 samples across all instruments (2000-2025)

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

### 1. Multi-Instrument Data Fetcher
- Retrieves daily OHLC data for NQ, ES, and YM futures via yfinance/stooq
- Handles multiple instruments with consistent data quality across all sources
- Key methods:
  - `fetch_multi_instrument_data(instruments, start_date, end_date)`: Retrieve data for all instruments
  - `calculate_previous_day_levels(date)`: Calculate previous day's high and low
  - `validate_data_quality(instrument_data)`: Ensure consistent data across instruments

### 2. Pure Visual Chart Generator
- Creates clean training images showing 30 daily bars with previous day level lines
- Draws previous day high as green horizontal line and previous day low as red horizontal line
- Generates charts optimized for pure visual pattern recognition (224x224 pixels)
- Key methods:
  - `create_pure_visual_chart(data, prev_high, prev_low)`: Create clean chart with color-coded levels
  - `save_chart_image(image, filename)`: Save chart image for training dataset
  - **No numerical overlays**: Pure visual information only

### 3. Daily Pattern Analyzer
- Analyzes single trading day to determine pattern classification
- Compares daily high/low against previous day levels
- Determines pattern type based on level interactions (same logic across all instruments)
- Key methods:
  - `analyze_daily_pattern(daily_data, prev_high, prev_low)`: Determine 4-class pattern
  - `classify_pattern(high_touched, low_touched)`: Assign pattern label 1-4

### 4. Multi-Instrument Dataset Creator
- Orchestrates the complete multi-instrument data generation workflow
- Creates training samples with pure visual charts and pattern labels
- Manages dataset organization for pure visual model training
- Key methods:
  - `generate_multi_instrument_dataset(instruments, start_date, end_date)`: Main workflow orchestrator
  - `create_visual_sample(chart, label, instrument)`: Generate sample with image + label + instrument tag
  - `export_pure_visual_dataset()`: Save complete dataset for pure visual model training

## Key Technical Requirements

### Data Sources
- **yfinance/stooq**: Multi-instrument data source for NQ, ES, YM futures
- **mplfinance**: Chart generation for pure visual model input

### Pure Visual Analysis Framework
- **Chart Timeframe**: Daily bars
- **Chart Duration**: 30 daily bars (approximately 6 weeks context) for each training image
- **Pattern Analysis**: Compare daily high/low against previous day levels (visual only)
- **Previous Day Calculation**: High and low from prior trading day
- **Visual Elements**: Clean candlestick charts with green (prev high) and red (prev low) reference lines
- **No Numerical Features**: Pure visual learning, no numerical overlays or annotations

### Multi-Instrument Data Splits (Strict Separation)
- **Training Set**: 2000-2020 (20 years historical data across all instruments)
- **Validation Set**: 2021-2022 (2 years model tuning across all instruments)
- **Test Set**: 2023-2025 (3 years final evaluation across all instruments)
- **Total Dataset**: ~20,000 samples spanning 25 years (2000-2025) across 3 instruments
- **No Data Leakage**: Strict temporal separation prevents look-ahead bias

### Pure Visual Classification Logic
1. **Chart Generation**: Create 30-bar daily chart with color-coded previous day levels (green=high, red=low)
2. **Pure Visual Input**: No numerical features - chart contains all necessary information visually
3. **Pattern Classification** (4-class system):
   - **Label 1**: High Breakout (daily high >= previous day high only)
   - **Label 2**: Low Breakdown (daily low <= previous day low only)
   - **Label 3**: Range Expansion (both levels touched during the day)
   - **Label 4**: Range Bound (neither level touched during the day)

4. **Training Sample**:
   - **Image**: 30-bar daily chart with color-coded previous day levels (224x224 pixels)
   - **Label**: Pattern classification (1-4)
   - **Instrument**: Instrument identifier (NQ, ES, or YM)
   - **No Numerical Features**: Pure visual learning approach

**Example**: Daily high reaches 15,105 (previous day high was 15,100) but daily low stays at 15,020 (previous day low was 15,000). Classification = Label 1 (High Breakout).

## Project Structure

```
/src
  daily_data_fetcher.py        # Multi-instrument data fetching (NQ, ES, YM)
  daily_chart_generator.py     # Pure visual chart generation (224x224)
  daily_pattern_analyzer.py    # Pattern classification logic
  daily_dataset_creator.py     # Multi-instrument dataset creation
  pure_visual_vit_model.py     # Pure visual ViT-Base (87M params)
  train_pure_visual.py         # Pure visual training pipeline
  generate_multi_dataset.py    # Single-command multi-instrument generation
  daily_data_loader.py         # Load multi-instrument dataset for training
  # Legacy hybrid models (for reference)
  hybrid_vit_model.py          # Custom hybrid ViT (3.49M params) - LEGACY
  vit_base_hybrid_model.py     # ViT-Base hybrid (87M params) - LEGACY
/data
  /images                      # Generated daily chart images (all instruments)
  /labels                      # Daily pattern labels and metadata
  /metadata                    # Dataset summaries and manifests
/models
  /outputs                     # Model training artifacts and results
/config
  config_pure_visual.yaml      # Pure visual ViT-Base configuration
  # Legacy configurations (for reference)
  config.yaml                  # Custom hybrid ViT configuration - LEGACY
  config_vit_base.yaml         # ViT-Base hybrid configuration - LEGACY
```

## Model Architecture

### Pure Visual ViT-Base (87M parameters)  
- Google ViT-Base-Patch16-224 architecture
- 12 transformer layers, 768 hidden size, 12 attention heads
- **Pure visual input**: 224x224 chart images only
- **No numerical features**: Eliminates fusion complexity
- **Multi-instrument training**: Learns patterns across NQ, ES, YM
- Optimized for visual pattern recognition on financial charts

## Development Priorities

1. **Multi-Instrument Dataset**: ~20,000 samples across NQ, ES, YM (2000-2025) 🚧 IN PROGRESS
2. **Pure Visual ViT-Base**: 87M parameter pure visual model 🚧 IN PROGRESS
3. **Training Pipeline**: Streamlined pure visual training 🚧 IN PROGRESS
4. **Performance Validation**: Compare against previous hybrid approaches ⏳ PENDING

## Important Implementation Notes

- **Chart Display**: Show 30 daily bars (approximately 6 weeks) of price action
- **Level Visualization**: Draw previous day high (green) and low (red) as horizontal lines
- **Pattern Analysis**: Compare current day high/low against previous day levels
- **Pure Visual Input**: Only chart images (224x224), no numerical features
- **Multi-Instrument**: Same pattern logic applied across NQ, ES, YM futures
- **No Data Augmentation**: Preserve chart integrity for financial data
- **Simple Logic**: Focus only on previous day level interactions for pattern classification

## Current Status

✅ **Legacy Hybrid Models**: Completed but underperformed (38% accuracy with numerical features)
🚧 **Pure Visual Architecture**: Implementing ViT-Base for pure visual learning
🚧 **Multi-Instrument Pipeline**: Building NQ + ES + YM dataset generation
⏳ **Training Pipeline**: Ready for RunPod deployment upon completion

## Usage

```bash
# Generate multi-instrument dataset (NQ, ES, YM)
python src/generate_multi_dataset.py

# Train pure visual ViT-Base model
python src/train_pure_visual.py --config config_pure_visual.yaml

# Legacy hybrid training (reference only)
python src/train_vit_base.py --config config_vit_base.yaml
```

