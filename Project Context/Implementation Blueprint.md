# Implementation Blueprint: NQ_AI Complete Technical Specification

This document contains the complete, unambiguous technical specification for implementing the NQ_AI system in one shot. Every detail, edge case, and implementation decision is documented here.

## 1. System Overview

The NQ_AI system is a specialized trading analysis tool that uses hindsight methodology to train an AI model on NASDAQ-100 E-mini futures (/NQ) patterns. It identifies which swing levels are likely to be swept before major structural breakouts.

## 2. Market Structure State Machine

### 2.1 Three Market States

```python
class MarketState(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    UNDEFINED = "UNDEFINED"  # When single candle sweeps entire range
```

### 2.2 State Transition Rules

#### UPTREND State
- **Entry**: Price breaks above structural_high
- **Maintenance**: Series of higher structural highs and higher structural lows
- **Exit to DOWNTREND**: Price trades below structural_low (Change of Character)
- **Exit to UNDEFINED**: Single candle sweeps both structural_high and structural_low

#### DOWNTREND State
- **Entry**: Price breaks below structural_low
- **Maintenance**: Series of lower structural lows and lower structural highs
- **Exit to UPTREND**: Price trades above structural_high (Change of Character)
- **Exit to UNDEFINED**: Single candle sweeps both structural_high and structural_low

#### UNDEFINED State
- **Entry**: Single candle trades through entire structural range
- **The candle's high becomes temporary structural_high**
- **The candle's low becomes temporary structural_low**
- **Exit to UPTREND**: Price breaks above the candle's high
- **Exit to DOWNTREND**: Price breaks below the candle's low

### 2.3 Structural Level Creation Rules

#### In UPTREND
1. **New Structural High**: When price breaks above current structural_high, the FIRST swing high formed after the break (first candle whose low is broken) becomes the new structural_high
2. **New Structural Low**: The swing low (pullback) is promoted to structural_low ONLY when price breaks above the most recent structural_high (Continuation Confirmation)

#### In DOWNTREND
1. **New Structural Low**: When price breaks below current structural_low, the FIRST swing low formed after the break (first candle whose high is broken) becomes the new structural_low
2. **New Structural High**: The swing high (pullback) is promoted to structural_high ONLY when price breaks below the most recent structural_low (Continuation Confirmation)

## 3. Swing Detection Mechanics

### 3.1 Swing Formation Rules

**Swing High Formation**: Occurs when a candle's low breaks below the previous candle's low
- The previous candle's high becomes the swing high
- Only tracked AFTER structural levels are established

**Swing Low Formation**: Occurs when a candle's high breaks above the previous candle's high
- The previous candle's low becomes the swing low
- Only tracked AFTER structural levels are established

### 3.2 Swing Tracking Timeline

1. After a structural break is confirmed, begin looking for swings
2. The FIRST swing in the direction of the break becomes the new structural level
3. Continue tracking all swings within the structural range
4. Apply unswept filter when preparing data for AI

### 3.3 Unswept Filter Algorithm

```python
def apply_unswept_filter(swing_levels, current_timestamp):
    """
    Keep only the 'outermost' swing levels that haven't been swept
    """
    valid_swing_highs = []
    valid_swing_lows = []
    
    # Sort swings by timestamp (oldest first)
    sorted_highs = sorted(swing_highs, key=lambda x: x.timestamp)
    sorted_lows = sorted(swing_lows, key=lambda x: x.timestamp)
    
    # For swing highs: keep if no newer swing high is higher
    for i, sh in enumerate(sorted_highs):
        is_valid = True
        for j in range(i+1, len(sorted_highs)):
            if sorted_highs[j].price >= sh.price:
                is_valid = False
                break
        if is_valid:
            valid_swing_highs.append(sh)
    
    # For swing lows: keep if no newer swing low is lower
    for i, sl in enumerate(sorted_lows):
        is_valid = True
        for j in range(i+1, len(sorted_lows)):
            if sorted_lows[j].price <= sl.price:
                is_valid = False
                break
        if is_valid:
            valid_swing_lows.append(sl)
    
    return valid_swing_highs, valid_swing_lows
```

## 4. Data Pipeline Architecture

### 4.1 Data Source Configuration

```yaml
data_source:
  ticker: "NQ=F"  # Continuous futures contract
  raw_interval: "1h"  # Download hourly data from yfinance
  target_interval: "4h"  # Aggregate to 4-hour bars
  timezone: "America/New_York"  # All times in EST
```

### 4.2 Data Aggregation Method

```python
def aggregate_1h_to_4h(hourly_data):
    """
    Aggregate 1-hour bars to 4-hour bars
    Groups: [0-3], [4-7], [8-11], [12-15], [16-19], [20-23] hours
    """
    # Ensure timezone is EST
    hourly_data.index = hourly_data.index.tz_convert('America/New_York')
    
    # Define 4-hour groups starting at midnight
    four_hour_groups = {
        0: [0, 1, 2, 3],
        4: [4, 5, 6, 7],
        8: [8, 9, 10, 11],
        12: [12, 13, 14, 15],
        16: [16, 17, 18, 19],
        20: [20, 21, 22, 23]
    }
    
    aggregated = hourly_data.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    return aggregated
```

### 4.3 Historical Data Retrieval

```python
def fetch_historical_data():
    """
    Fetch NQ futures data from yfinance
    """
    import yfinance as yf
    
    # Download maximum available history
    ticker = yf.Ticker("NQ=F")
    
    # Get 1-hour data (yfinance doesn't support 4h directly)
    hourly_data = ticker.history(period="max", interval="1h")
    
    # If data before 2018 isn't available, try specific date range
    if hourly_data.index[0].year > 2018:
        hourly_data = ticker.history(
            start="2018-01-01",
            end="2025-12-31",
            interval="1h"
        )
    
    # Aggregate to 4-hour bars
    four_hour_data = aggregate_1h_to_4h(hourly_data)
    
    return four_hour_data
```

## 5. Session Classification Logic

### 5.1 Session Time Windows

```python
SESSION_START = time(8, 0)  # 8:00 AM EST
SESSION_END = time(17, 0)   # 5:00 PM EST
MARKET_OPEN = time(9, 30)   # 9:30 AM EST
```

### 5.2 Session Outcome Types

```python
class SessionOutcome(Enum):
    BREAKOUT_UP = "BREAKOUT_UP"      # Structural high broken during session
    BREAKOUT_DOWN = "BREAKOUT_DOWN"  # Structural low broken during session
    RANGEBOUND = "RANGEBOUND"        # No structural breaks during session
```

### 5.3 Classification Algorithm

```python
def classify_session(session_data, structural_levels):
    """
    Classify a trading session (8:00-17:00 EST)
    """
    session_high = session_data['High'].max()
    session_low = session_data['Low'].min()
    
    # Check for structural breaks
    if session_high > structural_levels['structural_high']:
        return SessionOutcome.BREAKOUT_UP
    elif session_low < structural_levels['structural_low']:
        return SessionOutcome.BREAKOUT_DOWN
    else:
        return SessionOutcome.RANGEBOUND
```

### 5.4 Catalyst Identification

```python
def find_catalyst(breakout_timestamp, swing_levels, direction):
    """
    Find the last swept swing before structural break
    """
    # Get all swings before breakout
    prior_swings = [s for s in swing_levels if s.timestamp < breakout_timestamp]
    
    if direction == "UP":
        # Find last swing low that was swept
        swept_lows = [sl for sl in prior_swings 
                      if sl.type == "LOW" and 
                      price_traded_below(sl.price, before=breakout_timestamp)]
        if swept_lows:
            return max(swept_lows, key=lambda x: x.timestamp)
    
    elif direction == "DOWN":
        # Find last swing high that was swept
        swept_highs = [sh for sh in prior_swings 
                       if sh.type == "HIGH" and 
                       price_traded_above(sh.price, before=breakout_timestamp)]
        if swept_highs:
            return max(swept_highs, key=lambda x: x.timestamp)
    
    return None
```

## 6. Chart Generation Specifications

### 6.1 Image Parameters

```python
CHART_CONFIG = {
    'width': 1920,
    'height': 1080,
    'dpi': 100,
    'bars_to_display': 150,  # ~30 days of 4-hour bars
    'style': 'charles',       # Clean candlestick style
    'volume': False,          # No volume bars
    'ylabel': 'Price',
    'datetime_format': '%Y-%m-%d %H:%M'
}
```

### 6.2 Chart Generation Function

```python
def generate_chart_image(data, timestamp, output_path):
    """
    Generate clean candlestick chart for AI training
    """
    import mplfinance as mpf
    
    # Get 150 bars ending at timestamp
    end_idx = data.index.get_loc(timestamp)
    start_idx = max(0, end_idx - 150)
    chart_data = data.iloc[start_idx:end_idx+1]
    
    # Create clean chart (no annotations)
    mpf.plot(
        chart_data,
        type='candle',
        style='charles',
        volume=False,
        ylabel='Price',
        savefig=dict(
            fname=output_path,
            dpi=100,
            bbox_inches='tight',
            pad_inches=0.1
        ),
        figsize=(19.2, 10.8),  # 1920x1080 at 100 DPI
        returnfig=False
    )
```

## 7. Feature Engineering

### 7.1 Feature Set ("Battlefield Map")

For each swing level candidate, calculate:

```python
def calculate_features(candidate_level, all_levels, current_price, structural_range):
    """
    Generate complete feature set for a swing level
    """
    features = {}
    
    # Basic features
    features['distance_from_price'] = abs(candidate_level.price - current_price)
    features['normalized_distance'] = features['distance_from_price'] / structural_range
    features['age_in_bars'] = current_bar_index - candidate_level.bar_index
    features['position_in_range'] = (candidate_level.price - structural_low) / structural_range
    
    # Relative distances to ALL other levels
    for other_level in all_levels:
        if other_level.id != candidate_level.id:
            key = f"distance_to_{other_level.type}_{other_level.id}"
            features[key] = abs(candidate_level.price - other_level.price) / structural_range
    
    # Distances to structural levels
    features['distance_to_structural_high'] = abs(candidate_level.price - structural_high) / structural_range
    features['distance_to_structural_low'] = abs(candidate_level.price - structural_low) / structural_range
    
    # Previous structural levels (if exist)
    if previous_structural_high:
        features['distance_to_prev_structural_high'] = abs(candidate_level.price - previous_structural_high) / structural_range
    if previous_structural_low:
        features['distance_to_prev_structural_low'] = abs(candidate_level.price - previous_structural_low) / structural_range
    
    return features
```

### 7.2 Normalization Strategy

- **Price distances**: Normalize by structural range size (structural_high - structural_low)
- **Time features**: Normalize by maximum lookback period (150 bars)
- **Relative positions**: Already bounded 0-1 by definition

## 8. Market Structure Engine Implementation

### 8.1 Initialization

```python
class MarketStructureEngine:
    def __init__(self, ohlc_data):
        self.data = ohlc_data
        self.market_state = None
        self.structural_high = None
        self.structural_low = None
        self.previous_structural_high = None
        self.previous_structural_low = None
        self.swing_highs = []
        self.swing_lows = []
        self.history = []
        
        # Initialize from first 150 bars
        self._initialize_structure()
    
    def _initialize_structure(self):
        """
        Find initial swing high/low to establish first structural range
        """
        lookback = min(150, len(self.data))
        
        # Find first swing high and low
        for i in range(1, lookback):
            current = self.data.iloc[i]
            previous = self.data.iloc[i-1]
            
            # Check for swing high
            if current['Low'] < previous['Low'] and not self.structural_high:
                self.structural_high = previous['High']
                
            # Check for swing low  
            if current['High'] > previous['High'] and not self.structural_low:
                self.structural_low = previous['Low']
            
            # Once we have both, determine initial state
            if self.structural_high and self.structural_low:
                # The most recent break determines state
                if i > 0:
                    last_price = self.data.iloc[i-1]['Close']
                    mid_point = (self.structural_high + self.structural_low) / 2
                    self.market_state = MarketState.UPTREND if last_price > mid_point else MarketState.DOWNTREND
                break
```

### 8.2 Main Analysis Loop

```python
def run_analysis(self):
    """
    Process each bar and update market structure
    """
    start_index = 150  # Start after initialization
    
    for i in range(start_index, len(self.data)):
        current_bar = self.data.iloc[i]
        previous_bar = self.data.iloc[i-1]
        
        # Check for range sweep (UNDEFINED state)
        if self._check_range_sweep(current_bar):
            self._handle_range_sweep(current_bar, i)
            continue
        
        # Apply state-specific logic
        if self.market_state == MarketState.UPTREND:
            self._process_uptrend(current_bar, previous_bar, i)
        elif self.market_state == MarketState.DOWNTREND:
            self._process_downtrend(current_bar, previous_bar, i)
        elif self.market_state == MarketState.UNDEFINED:
            self._process_undefined(current_bar, i)
        
        # Log state for history
        self._log_state(i)
```

## 9. Data Factory Implementation

### 9.1 Main Dataset Generation

```python
class DataFactory:
    def __init__(self, market_engine, raw_data):
        self.engine = market_engine
        self.raw_data = raw_data
        self.labeled_samples = []
    
    def generate_dataset(self, start_date="2018-01-01", end_date="2022-12-31"):
        """
        Generate labeled dataset using hindsight methodology
        """
        trading_days = pd.bdate_range(start=start_date, end=end_date)
        
        for day in trading_days:
            # Define session window (8:00-17:00 EST)
            session_start = pd.Timestamp(day).replace(hour=8, minute=0)
            session_end = pd.Timestamp(day).replace(hour=17, minute=0)
            
            # Get session data
            session_data = self.raw_data[session_start:session_end]
            if len(session_data) == 0:
                continue
            
            # Get pre-market timestamp for image
            premarket_time = pd.Timestamp(day).replace(hour=8, minute=0)
            
            # Get market structure at premarket
            structure = self.engine.get_structure_at_timestamp(premarket_time)
            
            # Classify session outcome
            outcome = self._classify_session(session_data, structure)
            
            # Generate chart image
            image_path = self._generate_chart(premarket_time, day)
            
            # Get all unswept levels
            valid_levels = self._get_unswept_levels(structure, premarket_time)
            
            if outcome == SessionOutcome.RANGEBOUND:
                # Single sample for rangebound day
                features = self._calculate_all_features(valid_levels, structure, premarket_time)
                self.labeled_samples.append({
                    'date': day,
                    'image': image_path,
                    'features': features,
                    'label': 'RANGEBOUND'
                })
            else:
                # Find catalyst
                catalyst = self._find_catalyst(session_data, structure, outcome)
                
                # Create samples for each candidate
                for level in valid_levels:
                    features = self._calculate_features(level, valid_levels, structure, premarket_time)
                    label = 1 if level == catalyst else 0
                    
                    self.labeled_samples.append({
                        'date': day,
                        'image': image_path,
                        'features': features,
                        'candidate_level': level.price,
                        'label': label
                    })
```

## 10. Edge Cases and Error Handling

### 10.1 Gap Handling
- Gaps are treated as continuous price movement
- If gap crosses structural level, treat as definitive break
- No special gap detection needed

### 10.2 No Swings Scenario
- Guaranteed at least one swing in each structural range
- Uptrend: At least one swing low (becomes structural low on continuation)
- Downtrend: At least one swing high (becomes structural high on continuation)

### 10.3 Data Availability Issues
```python
def handle_data_gaps(data):
    """
    Handle missing data or insufficient history
    """
    # Forward fill gaps up to 1 day
    data = data.ffill(limit=6)  # 6 four-hour bars = 1 day
    
    # Drop remaining NaN values
    data = data.dropna()
    
    # Ensure minimum data for initialization
    if len(data) < 150:
        raise ValueError(f"Insufficient data: {len(data)} bars, need at least 150")
    
    return data
```

### 10.4 Timezone Handling
```python
def ensure_est_timezone(data):
    """
    Ensure all timestamps are in EST
    """
    if data.index.tz is None:
        # Assume UTC if no timezone
        data.index = data.index.tz_localize('UTC')
    
    # Convert to EST
    data.index = data.index.tz_convert('America/New_York')
    
    return data
```

## 11. Validation and Testing

### 11.1 Validation Criteria
- Each trading day must have a classification
- All breakout days must have an identified catalyst
- Images must contain exactly 150 bars (or less if at data start)
- Features must be normalized between reasonable bounds

### 11.2 Test Cases
```python
test_scenarios = [
    "Normal uptrend with multiple swings",
    "Downtrend with immediate reversal",
    "Range sweep creating UNDEFINED state",
    "Multiple rangebound days in sequence",
    "Gap up breaking structural high",
    "Gap down breaking structural low",
    "Session with no price movement",
    "First day of available data"
]
```

### 11.3 Success Metrics
- Market structure transitions match manual analysis
- Catalyst identification accuracy > 95%
- Image generation without errors for all days
- Feature normalization keeps values in [0, 10] range

## 12. Configuration File (config.yaml)

```yaml
# NQ_AI Configuration File
data:
  ticker: "NQ=F"
  raw_interval: "1h"
  target_interval: "4h"
  timezone: "America/New_York"
  
dataset:
  train_start: "2018-01-01"
  train_end: "2022-12-31"
  val_start: "2023-01-01"
  val_end: "2023-12-31"
  test_start: "2024-01-01"
  
session:
  premarket_time: "08:00"
  session_start: "08:00"
  session_end: "17:00"
  market_open: "09:30"
  
market_structure:
  initialization_bars: 150
  
charts:
  width: 1920
  height: 1080
  dpi: 100
  bars_to_display: 150
  style: "charles"
  
paths:
  data_dir: "./data"
  raw_data: "./data/raw"
  processed_data: "./data/processed"
  images_dir: "./data/images"
  models_dir: "./models"
  
logging:
  level: "INFO"
  file: "./logs/nq_ai.log"
```

## 13. Implementation Checklist

- [ ] Set up project structure
- [ ] Create config.yaml from template
- [ ] Implement data fetcher with 1h→4h aggregation
- [ ] Build MarketStructureEngine with three states
- [ ] Implement swing detection logic
- [ ] Create unswept filter algorithm
- [ ] Build DataFactory with hindsight methodology
- [ ] Implement session classification
- [ ] Create chart generation pipeline
- [ ] Build feature engineering system
- [ ] Add comprehensive logging
- [ ] Create validation notebooks
- [ ] Test with historical data
- [ ] Verify catalyst identification
- [ ] Validate image generation
- [ ] Check feature normalization

## 14. Final Notes

This specification is complete and unambiguous. Every decision has been made, every edge case considered. The implementation should follow this blueprint exactly to create a functional NQ_AI system that:

1. Correctly tracks market structure through all three states
2. Identifies swings using mechanical rules only
3. Filters to unswept levels appropriately
4. Classifies sessions accurately
5. Generates clean training images
6. Produces normalized features for AI training
7. Handles all edge cases gracefully

With this blueprint, the implementation can proceed without any further clarification needed.