# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NQ_AI is an AI trading system for NASDAQ-100 E-mini futures (/NQ) that analyzes daily price patterns using previous day levels. The system generates training data by capturing daily chart images with 30 bars of context and analyzing price action patterns.

**Core Function**: Four-class classification system that predicts daily patterns:
- **1**: High Breakout (Daily high >= previous day high only)
- **2**: Low Breakdown (Daily low <= previous day low only)  
- **3**: Range Expansion (Both levels touched during the day)
- **4**: Range Bound (Neither level touched during the day)

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

The system implements a daily pattern analysis pipeline with four core components:

### 1. Daily Data Fetcher
- Retrieves daily OHLC data for /NQ futures via yfinance
- Generates chart images showing 30 daily bars with previous day levels
- Key methods:
  - `fetch_daily_data(start_date, end_date)`: Retrieve historical daily data
  - `calculate_previous_day_levels(date)`: Calculate previous day's high and low
  - `generate_numerical_features()`: Create distance and range features

### 2. Daily Chart Generator
- Creates training images showing 30 daily bars with previous day level lines
- Draws previous day high as green horizontal line and previous day low as red horizontal line
- Generates clean charts optimized for daily pattern recognition
- Key methods:
  - `create_daily_chart(data, prev_high, prev_low)`: Create chart with color-coded levels
  - `save_chart_image(image, filename)`: Save chart image for training dataset

### 3. Daily Pattern Analyzer
- Analyzes single trading day to determine pattern classification
- Compares daily high/low against previous day levels
- Determines pattern type based on level interactions
- Key methods:
  - `analyze_daily_pattern(daily_data, prev_high, prev_low)`: Determine 4-class pattern
  - `classify_pattern(high_touched, low_touched)`: Assign pattern label 1-4

### 4. Daily Dataset Creator
- Orchestrates the complete daily data generation workflow
- Creates training samples with chart images, numerical features, and pattern labels
- Manages dataset organization for hybrid model training
- Key methods:
  - `generate_daily_dataset(start_date, end_date)`: Main workflow orchestrator
  - `create_hybrid_sample(chart, features, label)`: Generate sample with images + features
  - `export_hybrid_dataset()`: Save complete dataset for hybrid model training

## Key Technical Requirements

### Data Sources
- **yfinance**: Primary source for /NQ futures daily data
- **mplfinance**: Chart generation for visual model input

### Daily Analysis Framework
- **Chart Timeframe**: Daily bars
- **Chart Duration**: 30 daily bars (approximately 6 weeks context) for each training image
- **Pattern Analysis**: Compare daily high/low against previous day levels
- **Previous Day Calculation**: High and low from prior trading day

### Data Splits (Strict Separation)
- **Training Set**: 2000-2020 (20 years historical data)
- **Validation Set**: 2021-2022 (2 years model tuning)
- **Test Set**: 2023-2025 (3 years final evaluation)
- **Total Dataset**: 6,235 samples spanning 25 years (2000-2025)
- **No Data Leakage**: Strict temporal separation prevents look-ahead bias

### Pattern Classification Logic
1. **Chart Generation**: Create 30-bar daily chart with color-coded previous day levels (green=high, red=low)
2. **Feature Extraction**: Calculate 3 numerical features:
   - Distance to previous high (prev_high - open_price)
   - Distance to previous low (open_price - prev_low)  
   - Previous day range (prev_high - prev_low)
3. **Pattern Classification** (4-class system):
   - **Label 1**: High Breakout (daily high >= previous day high only)
   - **Label 2**: Low Breakdown (daily low <= previous day low only)
   - **Label 3**: Range Expansion (both levels touched during the day)
   - **Label 4**: Range Bound (neither level touched during the day)

4. **Training Sample**:
   - **Image**: 30-bar daily chart with color-coded previous day levels
   - **Features**: 3 numerical features for additional context
   - **Label**: Pattern classification (1-4)

**Example**: Daily high reaches 15,105 (previous day high was 15,100) but daily low stays at 15,020 (previous day low was 15,000). Classification = Label 1 (High Breakout).

## Project Structure

```
/src
  daily_data_fetcher.py     # Fetch daily NQ futures data
  daily_chart_generator.py  # Create 30-bar daily charts with levels
  daily_pattern_analyzer.py # Analyze daily patterns for classification
  daily_dataset_creator.py  # Generate complete daily dataset
  hybrid_vit_model.py       # Custom hybrid ViT (3.49M params)
  vit_base_hybrid_model.py  # ViT-Base hybrid (87M params)
  daily_data_loader.py      # Load daily dataset for training
  train_daily_model.py      # Train custom hybrid ViT
  train_vit_base.py         # Train ViT-Base hybrid
/data
  /images                   # Generated daily chart images
  /labels                   # Daily pattern labels and metadata
  /metadata                 # Dataset summaries and manifests
/models
  /outputs                  # Model training artifacts and results
/config
  config.yaml               # Custom hybrid ViT configuration
  config_vit_base.yaml      # ViT-Base hybrid configuration
```

## Model Architectures

### 1. Custom Hybrid ViT (3.49M parameters)
- 6 transformer blocks, 256 projection dim, 8 attention heads
- Combines chart images + 3 numerical features
- Optimized for daily pattern recognition
- Fast training and inference

### 2. ViT-Base Hybrid (87M parameters)  
- Google ViT-Base-Patch16-224 architecture
- 12 transformer layers, 768 hidden size, 12 attention heads
- Combines chart images + 3 numerical features
- Higher capacity for complex pattern learning
- Longer training time, higher computational requirements

## Development Priorities

1. **Daily Dataset**: 6,235 samples of daily patterns (2000-2025) ✅ COMPLETED
2. **Custom Hybrid ViT**: Efficient 3.49M parameter model ✅ COMPLETED
3. **ViT-Base Hybrid**: High-capacity 87M parameter model (IN PROGRESS)
4. **Model Comparison**: Performance analysis of both architectures

## Important Implementation Notes

- **Chart Display**: Show 30 daily bars (approximately 6 weeks) of price action
- **Level Visualization**: Draw previous day high (green) and low (red) as horizontal lines
- **Pattern Analysis**: Compare current day high/low against previous day levels
- **Hybrid Input**: Both models use chart images + 3 numerical features
- **Simple Logic**: Focus only on previous day level interactions for pattern classification

## Current Status

✅ **Daily Dataset Generation**: Complete 25-year dataset with 6,235 samples (2000-2025)
✅ **Custom Hybrid ViT**: 3.49M parameter model implemented and training ready
🚧 **ViT-Base Hybrid**: 87M parameter model implementation in progress
⏳ **Model Comparison**: Performance evaluation pending completion of both models

## Usage

```bash
# Train custom hybrid ViT (3.49M params)
python src/train_daily_model.py

# Train ViT-Base hybrid (87M params) 
python src/train_vit_base.py --model base

# Model selection via config
python src/train_vit_base.py --config config_vit_base.yaml
```

