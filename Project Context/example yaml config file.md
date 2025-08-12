# ===================================================================
# EXAMPLE CONFIGURATION FILE for the NQ AI Project
# ===================================================================
# This file contains examples of all the key parameters for our data generation
# and analysis scripts.

# --- 1. Data Acquisition & Timeframe Settings ---
data_source:
  ticker: "/NQ=F"  # The primary ticker symbol for Nasdaq-100 E-mini futures.
  interval: "4h"    # The chart timeframe for analysis.

dataset_split:
  training_start: "2018-01-01"
  training_end: "2022-12-31"
  validation_start: "2023-01-01"
  validation_end: "2023-12-31"
  test_start: "2024-01-01"
  test_end: "2025-12-31" # Set to a future date to capture all recent data.

# --- 2. Session & Analysis Window ---
session:
  timezone: "America/New_York" # Use a single timezone to handle daylight saving correctly.
  snapshot_time: "08:00"      # Time to capture the pre-market chart image.
  session_start: "08:00"      # Start of the analysis window.
  session_end: "17:00"        # End of the analysis window (5 PM EST).

# --- 3. Market Structure Engine Settings ---
market_structure:
  initial_warmup_period: 150 # Number of initial bars to analyze to determine the first market state.

# --- 4. Data Factory & Image Generation Settings ---
data_factory:
  image_candles_to_show: 150 # Number of 4-hour candles to include in each training image. Approx. 30 days
  output_image_size: [1920, 1080] # Standardized image dimensions (width, height) for the AI model.
  output_image_dpi: 100 # Dots per inch, controls the resolution of the saved chart images.

# --- 5. AI Model Training Settings ---
# These parameters will be used when we get to the model training phase.
model:
  base_model: "VisionTransformer" # The architecture we plan to fine-tune.
  learning_rate: 0.001
  batch_size: 32 # Number of images to show the AI at once during training.
  epochs: 50 # Number of full passes over the entire training dataset.