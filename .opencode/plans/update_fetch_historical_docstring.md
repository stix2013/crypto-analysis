# Update fetch_historical Docstring for Accuracy

## Problem
The docstring for `fetch_historical` in `src/crypto_analysis/data/binance.py` incorrectly describes the behavior of the `start_time` parameter:

Current docstring line 208:
```
start_time: Optional start time to fetch from (default: fetch latest)
```

This is misleading because:
1. When `start_time=None` (the default), the method doesn't simply "fetch latest" - it uses a backward-fetching approach to get the most recent bars efficiently by fetching progressively older data
2. When `start_time` is provided, it fetches forward from that time
3. The behavior is actually quite different between the two cases, which the current docstring doesn't clarify

## Solution
Update the docstring to accurately describe the behavior for both cases:
- When start_time is None: Uses backward-fetching approach to get most recent bars
- When start_time is provided: Fetches forward from the specified time

## Changes Needed
In `src/crypto_analysis/data/binance.py`, update the docstring for the `fetch_historical` method (lines 199-211) to accurately describe the behavior.

## Updated Docstring
```python
    def fetch_historical(
        self,
        symbol: str,
        interval: Interval,
        bars: int = 1000,
        start_time: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data efficiently.

        When start_time is None (default), uses a backward-fetching approach to
        get the most recent bars efficiently by fetching progressively older data.
        When start_time is provided, fetches data forward from that timestamp.

        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            interval: Kline interval
            bars: Number of bars to fetch (default 1000)
            start_time: Optional start time to fetch from (default: fetch most recent bars)

        Returns:
            DataFrame with OHLCV data
```

## Implementation Plan
Simply replace the existing docstring with the updated version that accurately describes the behavior.

## Testing
After updating the docstring, verify:
1. The docstring renders correctly in IDEs and documentation tools
2. The description accurately matches the actual implementation behavior
3. No functional changes were made (this is documentation-only)
