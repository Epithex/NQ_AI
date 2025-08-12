# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NQ_AI ("AI Liquidity Analyst") is a specialized AI trading system for NASDAQ-100 E-mini futures (/NQ) that uses a hybrid visual and numerical approach to predict breakout success probabilities. The system analyzes pre-market conditions daily at 8 AM EST on the 4-hour chart (primary view), focusing on mechanical market structure identification and historical pattern recognition.

**Core Innovation**: Multi-class classification system that identifies:
- **Success (1)**: Swing levels that will be swept leading to structural breakouts
- **Failure (0)**: Swing levels that won't lead to successful breakouts  
- **Rangebound (distinct class)**: Days with no significant breakouts ("no trade" conditions) - requires multi-class classifier architecture

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
- **Central Script**: Main implementation in `data_factory.py` developed interactively
- **Version Control**: Git repository with `.gitignore` for venv and large data files

## High-Level Architecture

The system implements a unique "hindsight" training methodology with four core components:

### Market Structure Definitive Rules

#### In Confirmed UPTREND:
- **New Structural High Creation**: After current Structural High is broken, the very first Swing High formed is instantly promoted to new Structural High
- **New Structural Low Creation**: A Swing Low (pullback) is only promoted to Structural Low when price breaks above the most recent Structural High (Continuation Confirmation)
- **Termination (Change of Character)**: UPTREND terminates and flips to DOWNTREND the moment price breaks below current Structural Low
  - Retain structural_high as new structural_high for downtrend
  - Move broken structural_low to previous_structural_low
  - Wait for first valid Swing Low to form and set as new structural_low

#### In Confirmed DOWNTREND:
- **New Structural Low Creation**: After current Structural Low is broken, the very first Swing Low formed is instantly promoted to new Structural Low
- **New Structural High Creation**: A Swing High (pullback) is only promoted to Structural High when price breaks below the most recent Structural Low (Continuation Confirmation)
- **Termination (Change of Character)**: DOWNTREND terminates and flips to UPTREND the moment price breaks above current Structural High
  - Retain structural_low as new structural_low for uptrend
  - Move broken structural_high to previous_structural_high
  - Wait for first valid Swing High to form and set as new structural_high

### 1. Market Structure Engine (Class: MarketStructureEngine)
- Mechanical rule-based system for tracking trend states (UPTREND/DOWNTREND)
- Identifies swing highs/lows using definitive pivot rules (candle high breaks previous high for swing low, candle low breaks previous low for swing high)
- Maintains state variables: structural_high, structural_low, swing_high, swing_low, previous_structural_high/low
- Initial state determination: Analyze recent data to find last completed swing high/low, treat as first Structural Range
- Key methods:
  - `__init__(ohlc_data: pd.DataFrame)`: Initialize with price data and determine initial state
  - `run_analysis()`: Main loop for bar-by-bar analysis
  - `_check_uptrend_logic(index, candle)`: Apply uptrend rules including "Change of Character" and "Continuation Confirmation"
  - `_check_downtrend_logic(index, candle)`: Apply downtrend rules (mirror of uptrend logic)
  - `_find_swing_high(index)`: Detect swing high formation (candle low breaks previous low)
  - `_find_swing_low(index)`: Detect swing low formation (candle high breaks previous high)
  - `get_structure_at_timestamp(timestamp: pd.Timestamp) -> dict`: Retrieve historical state from history log

### 2. Data Factory (Class: DataFactory)
- Works backward from successful breakouts or rangebound days ("hindsight" methodology)
- Mines 5-day patterns leading to events
- Generates multi-class training data from 2018-2022 period
- For rangebound days: Creates single sample with full feature set but no specific candidate
- Key methods:
  - `generate_dataset(start_date, end_date)`: Main orchestrator for data generation workflow
  - `_find_session_outcome(session_data)`: Scan session data (8:00-17:00 EST) to identify if major Structural High/Low was broken or if it was a range day
  - `_find_catalyst(breakout_event)`: Work backward to locate the last swept swing level before major move
  - `_capture_setup_image(setup_timestamp)`: Save clean, unaltered pre-market chart image
  - `_get_all_key_levels(setup_timestamp)`: Find all visible structural and "unswept" swing levels (apply unswept filter logic)
  - `_generate_full_feature_set(all_levels, candidate)`: Create complete "battlefield map" with relative features for ALL levels simultaneously
  - `save_labels_to_csv()`: Export labeled dataset

### 3. Hybrid AI Model
- **Visual Branch**: CNN for analyzing 5-day candlestick charts (4-hour timeframe)
- **Numerical Branch**: Dense network for relative features of ALL key levels provided simultaneously (complete context)
- **Fusion Layer**: Combines both branches for multi-class output
- **Output Classes**: Multi-class classifier with Success (1), Failure (0), Rangebound (distinct class)
- TensorFlow/Keras implementation with custom training pipeline
- Model receives full "battlefield map" of all levels for each prediction

### 4. Live Deployment System
- Daily execution at 8 AM EST before market open (9:30 AM)
- Session window: 8:00-17:00 EST for analysis
- Analyzes current market structure and recent patterns
- Generates probability scores for all visible swing levels
- Outputs ranked pre-market analysis with confidence scores

## Key Technical Requirements

### Data Sources
- **yfinance**: Primary source for /NQ futures data
- **mplfinance**: Chart generation for visual model input (clean charts for AI, annotated charts for validation)
- **5-minute timeframe**: For intraday analysis and precise entry/exit timing
- **4-hour chart**: Primary analysis view for structural levels and AI predictions

### Time Windows & Data Splits

#### Trading Sessions
- **Pre-market Analysis**: 8:00 AM EST (before market open)
- **Session Window**: 8:00-17:00 EST (full analysis period)
- **Market Open**: 9:30 AM EST (must complete analysis before this)

#### Data Splits (Strict Separation)
- **Training Set**: 2018-2022 (historical pattern mining)
- **Validation Set**: 2023 (model tuning and selection)
- **Test Set**: 2024-Present (final performance evaluation)
- **No Data Leakage**: Strict temporal separation prevents look-ahead bias

### Model Training Approach
1. Historical pattern identification from training set
2. Backward-looking data generation from successful breakouts
3. Dual-path neural network training with validation monitoring
4. Final evaluation on test set only

### Data Pipeline Details

#### Hindsight Methodology Workflow
1. **Find Outcome**: Scan historical data for successful structural breakouts or Session_Rangebound days
2. **Identify Catalyst**: For breakouts, work backward to find the last swept swing level (the Actual Catalyst)
3. **Capture Setup**: Save clean, unaltered pre-market chart image from event day (8:00 AM EST)
4. **Identify All Candidates**: Find all visible internal, unswept swing highs and lows at pre-market time
5. **Filter Candidates**: Apply "unswept" logic to swing levels:
   - Discard swing lows if newer, lower swing lows exist (already swept)
   - Discard swing highs if newer, higher swing highs exist (already swept)
6. **Generate Labeled Samples**:
   - Sample with Actual Catalyst level → Label: Success (1)
   - Samples with other candidate levels → Label: Failure (0)
   - Rangebound days → Single sample with full feature set → Label: Rangebound (distinct class)

#### Feature Engineering ("Battlefield Map")
For each swing level candidate, calculate complete relative feature set:
- `distance_from_current_price`: abs(level_price - current_price)
- `age_in_bars`: current_index - level_creation_index  
- `normalized_position_in_range`: (level_price - structural_low) / (structural_high - structural_low)
- Relative distances to ALL other key levels (structural, previous structural, all swing candidates)
- Complete context provided simultaneously to AI for each prediction
- Returns flat dictionary format: {'sh_distance': 50, 'sl_distance': 100, ...}

### Critical Implementation Rules
- **No subjective interpretation**: All market structure rules must be mechanical
- **Pre-market focus**: Analysis completes before 9:30 AM EST market open
- **Probability ranking**: AI scores all swing levels, not binary predictions
- **Historical validation**: Patterns must demonstrate statistical edge
- **Unswept filter**: Only consider swing levels that haven't been superseded

## Project Structure (To Be Implemented)

```
/src
  /market_structure    # Mechanical rule engine
  /data_factory       # Historical pattern mining
  /models             # Hybrid AI architecture
  /deployment         # Live execution system
  /utils              # Shared utilities
/notebooks            # Jupyter development notebooks
/data                 # Historical and live data storage
/models               # Trained model artifacts
/tests                # Unit and integration tests
```

## Blueprint Documentation

Complete system specification available in:
- `/Project Context/Blueprint/` - High-level strategic plan and core logic
- `/Project Context/Code Reference and Implementation/` - Detailed pseudocode and class schemas
- `/Project Context/Technical Blueprint/` - Complete technical schema and development guide
- Focus on "hindsight" methodology and definitive rule system
- Contains exact market structure definitions and AI architecture

## Development Priorities

1. **Market Structure Engine**: Implement mechanical rules first
2. **Data Factory**: Build historical pattern extraction
3. **Model Development**: Create and train hybrid architecture
4. **Live System**: Deploy daily pre-market analysis

## Important Conventions

- All market structure identification must use definitive, mechanical rules
- Pattern analysis works backward from known successful outcomes
- System focuses on probability assessment, not directional predictions
- Daily analysis timing is critical - must complete before market open