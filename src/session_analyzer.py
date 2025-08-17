# src/session_analyzer.py
import pandas as pd
from datetime import datetime, time
import pytz
from typing import Dict, List, Optional, Tuple
import yaml
import logging


class SessionAnalyzer:
    """Analyzes US trading session for level touch patterns."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize SessionAnalyzer with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.timezone = pytz.timezone(self.config["time"]["timezone"])
        self.session_start = time(8, 0)  # 8:00 AM EST
        self.session_end = time(17, 0)  # 5:00 PM EST

        # Setup logging
        logging.basicConfig(
            filename=f"{self.config['paths']['logs']}/session_analyzer.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def analyze_session(
        self, session_data: pd.DataFrame, prev_high: float, prev_low: float
    ) -> int:
        """
        Determine classification using priority order (5→6→1-4).

        Args:
            session_data: US session hourly data (8AM-5PM)
            prev_high: Previous day high
            prev_low: Previous day low

        Returns:
            Label classification (1-6)
        """
        try:
            self.logger.info(
                f"Analyzing session with PDH={prev_high:.2f}, PDL={prev_low:.2f}"
            )

            # Priority 1: Check if rangebound (Label 5)
            if self.is_rangebound(session_data, prev_high, prev_low):
                self.logger.info("Classification: Label 5 (Rangebound)")
                return 5

            # Priority 2: Check for simultaneous touches (Label 6)
            if self.detect_simultaneous_touches(session_data, prev_high, prev_low):
                self.logger.info("Classification: Label 6 (Simultaneous touches)")
                return 6

            # Priority 3: Analyze touch sequence for Labels 1-4
            label = self.track_level_touches(session_data, prev_high, prev_low)
            self.logger.info(f"Classification: Label {label}")
            return label

        except Exception as e:
            self.logger.error(f"Error analyzing session: {str(e)}")
            raise

    def is_rangebound(
        self, session_data: pd.DataFrame, prev_high: float, prev_low: float
    ) -> bool:
        """
        Check if neither level was touched during session (Label 5).

        Args:
            session_data: US session data
            prev_high: Previous day high
            prev_low: Previous day low

        Returns:
            True if rangebound (neither level touched)
        """
        high_touched = (session_data["High"] >= prev_high).any()
        low_touched = (session_data["Low"] <= prev_low).any()

        return not high_touched and not low_touched

    def detect_simultaneous_touches(
        self, session_data: pd.DataFrame, prev_high: float, prev_low: float
    ) -> bool:
        """
        Check if any single candle touches both levels (Label 6).

        Args:
            session_data: US session data
            prev_high: Previous day high
            prev_low: Previous day low

        Returns:
            True if any candle touches both levels
        """
        simultaneous = (session_data["High"] >= prev_high) & (
            session_data["Low"] <= prev_low
        )

        if simultaneous.any():
            touch_times = session_data[simultaneous].index
            self.logger.info(f"Simultaneous touches at: {touch_times.tolist()}")
            return True

        return False

    def track_level_touches(
        self, session_data: pd.DataFrame, prev_high: float, prev_low: float
    ) -> int:
        """
        Determine touch sequence for Labels 1-4.

        Args:
            session_data: US session data
            prev_high: Previous day high
            prev_low: Previous day low

        Returns:
            Label classification (1-4)
        """
        high_touches = session_data[session_data["High"] >= prev_high]
        low_touches = session_data[session_data["Low"] <= prev_low]

        high_touched = not high_touches.empty
        low_touched = not low_touches.empty

        # Label 1: Only touches green line (high)
        if high_touched and not low_touched:
            return 1

        # Label 2: Only touches red line (low)
        if low_touched and not high_touched:
            return 2

        # Both levels touched - determine sequence
        if high_touched and low_touched:
            first_high_time = high_touches.index[0]
            first_low_time = low_touches.index[0]

            # Label 3: Green first, then red
            if first_high_time < first_low_time:
                self.logger.info(
                    f"High touched first at {first_high_time}, then low at {first_low_time}"
                )
                return 3

            # Label 4: Red first, then green
            else:
                self.logger.info(
                    f"Low touched first at {first_low_time}, then high at {first_high_time}"
                )
                return 4

        # Should not reach here due to priority checks
        raise ValueError("Unable to classify session")

    def get_session_data(self, date: datetime, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract US trading session data for a specific date.

        Args:
            date: Trading date
            data: Full historical data

        Returns:
            DataFrame with session data (8AM-5PM EST)
        """
        # Ensure timezone-aware dates
        if date.tzinfo is None:
            date = self.timezone.localize(date)

        session_start = date.replace(hour=8, minute=0, second=0)
        session_end = date.replace(hour=17, minute=0, second=0)

        session_data = data.loc[session_start:session_end]

        if session_data.empty:
            raise ValueError(f"No session data for {date.date()}")

        self.logger.info(
            f"Session data: {len(session_data)} bars from {session_start} to {session_end}"
        )
        return session_data

    def determine_success_failure(self, actual_label: int) -> Dict[int, str]:
        """
        Create success/failure matrix for all labels.

        Args:
            actual_label: The actual classification (1-6)

        Returns:
            Dictionary with success/failure for each label
        """
        results = {}
        for label in range(1, 7):
            results[label] = "success" if label == actual_label else "failure"

        return results

    def get_touch_details(
        self, session_data: pd.DataFrame, prev_high: float, prev_low: float
    ) -> Dict:
        """
        Get detailed information about level touches.

        Args:
            session_data: US session data
            prev_high: Previous day high
            prev_low: Previous day low

        Returns:
            Dictionary with touch details
        """
        details = {"high_touches": [], "low_touches": [], "simultaneous_touches": []}

        for idx, row in session_data.iterrows():
            if row["High"] >= prev_high:
                details["high_touches"].append(
                    {
                        "time": idx,
                        "high": row["High"],
                        "excess": row["High"] - prev_high,
                    }
                )

            if row["Low"] <= prev_low:
                details["low_touches"].append(
                    {"time": idx, "low": row["Low"], "excess": prev_low - row["Low"]}
                )

            if row["High"] >= prev_high and row["Low"] <= prev_low:
                details["simultaneous_touches"].append(
                    {
                        "time": idx,
                        "high": row["High"],
                        "low": row["Low"],
                        "range": row["High"] - row["Low"],
                    }
                )

        return details
