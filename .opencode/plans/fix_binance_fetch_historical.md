# Fix BinanceClient.fetch_historical Inconsistent Behavior

## Problem
The `fetch_historical` method in `BinanceClient` behaves inconsistently when `start_time` is provided vs when it's None:
- When `start_time` is None: Uses progressive fetching to get the full requested number of bars
- When `start_time` is provided: Uses a single API request limited to 1500 bars max

This causes unexpected behavior where requesting 2000 bars with a start_time returns at most 1500 bars, while the same request without start_time returns ~2000 bars.

## Solution
Modify the `fetch_historical` method to use chunked fetching when `start_time` is provided, similar to the existing progressive fetching logic, ensuring consistent behavior regardless of whether start_time is specified.

## Changes Needed
1. In `src/crypto_analysis/data/binance.py`, replace the simple `_fetch_from_time` call when `start_time` is provided with a loop that fetches data in chunks of up to 1500 bars until the requested number of bars is obtained.

## Implementation Plan
```python
if start_time:
    start_ms = int(start_time.timestamp() * 1000)
    # Fetch in chunks if we need more than 1500 bars
    all_data = []
    remaining_bars = bars
    current_start_ms = start_ms

    while remaining_bars > 0:
        fetch_limit = min(remaining_bars, 1500)
        df_chunk = self._fetch_from_time(
            symbol, interval, current_start_ms, fetch_limit
        )

        if df_chunk.empty:
            break

        all_data.append(df_chunk)
        remaining_bars -= len(df_chunk)

        # Move start time forward for next chunk
        if len(df_chunk) < fetch_limit:
            break  # No more data available
        current_start_ms = int(df_chunk.index[-1].timestamp() * 1000) + 1

    if all_data:
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep="last")]
        return df.tail(bars)
    else:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
else:
    # Existing progressive fetching logic for recent data
    # ... (unchanged)
```

## Testing
After implementing the fix, verify:
1. `fetch_historical("BTCUSDT", "1h", 2000)` returns ~2000 bars
2. `fetch_historical("BTCUSDT", "1h", 2000, start_time=some_timestamp)` returns ~2000 bars
3. Edge cases like requesting fewer than 1500 bars work correctly
4. Edge cases where start_time is very far in the past work correctly
