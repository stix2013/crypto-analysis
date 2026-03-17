# Improve Backtester Price Data Storage Efficiency

## Problem
In `src/crypto_analysis/signals/backtest.py`, the `_update_data_handler` method (lines 84-89) uses `pd.concat` in a loop to build up price data:

```python
if timestamp not in self._price_data.index:
    new_row = pd.DataFrame({"close": [price]}, index=[timestamp])
    self._price_data = pd.concat([self._price_data, new_row])
```

This approach is inefficient because:
1. `pd.concat` creates a new DataFrame each time, copying all existing data
2. For N updates, this results in O(N²) time complexity
3. Memory usage grows unnecessarily due to repeated allocations

## Solution
Replace the inefficient `pd.concat` approach with one of these better alternatives:

### Option 1: Collect rows in a list, concatenate periodically
- Maintain a list of new rows
- When the list reaches a threshold (e.g., 100 rows), concatenate and reset
- This reduces the number of expensive concat operations

### Option 2: Pre-allocate DataFrame with expected size
- If we know the approximate number of updates, pre-allocate
- Use `.loc` to assign values by index
- Most efficient if we can predict the size

### Option 3: Use pandas DataFrame.append (deprecated but still available in some versions)
- Actually, `.append` is also inefficient for the same reasons as concat
- Not recommended

### Option 4: Use a different data structure
- Use a dict or list to collect data, convert to DataFrame only when needed
- Best for write-heavy scenarios

## Recommended Approach
Option 1 (collect in list, batch concatenate) provides the best balance of simplicity and performance improvement.

## Implementation Plan
Modify the `_update_data_handler` method in `src/crypto_analysis/signals/backtest.py`:

1. Add a `_price_data_rows` list attribute to collect new rows
2. Add a `_price_data_buffer_size` counter
3. When buffer reaches threshold (e.g., 100), concatenate and reset
4. Add a flush method to ensure all data is committed when needed

## Changes Needed
In `src/crypto_analysis/signals/backtest.py`:

1. Add attributes in `__init__`:
   ```python
   self._price_data_rows = []
   self._price_data_buffer_size = 0
   self._price_data_flush_threshold = 100
   ```

2. Modify `_update_data_handler` method:
   ```python
   if timestamp not in self._price_data.index:
       new_row = {"close": price}
       self._price_data_rows.append((timestamp, new_row))
       self._price_data_buffer_size += 1

       # Flush buffer if threshold reached
       if self._price_data_buffer_size >= self._price_data_flush_threshold:
           self._flush_price_data_buffer()
   ```

3. Add `_flush_price_data_buffer` method:
   ```python
   def _flush_price_data_buffer(self) -> None:
       """Flush buffered price data rows to main DataFrame."""
       if not self._price_data_rows:
           return

       # Convert buffered rows to DataFrame
       timestamps, rows = zip(*self._price_data_rows) if self._price_data_rows else ([], [])
       if timestamps:
           new_data = pd.DataFrame(list(rows), index=timestamps)
           if self._price_data is None:
               self._price_data = new_data
           else:
               self._price_data = pd.concat([self._price_data, new_data])
               self._price_data = self._price_data[~self._price_data.index.duplicated(keep="last")]

       # Clear buffer
       self._price_data_rows.clear()
       self._price_data_buffer_size = 0
   ```

4. Ensure buffer is flushed at appropriate times (e.g., in `get_equity_curve`, `get_trades`, `_calculate_results`)

## Testing
After implementing the fix, verify:
1. Functional correctness: price data is stored and retrieved correctly
2. Performance improvement: measure time for large numbers of updates
3. Edge cases: empty buffer, single item, exact threshold boundaries
4. Memory usage: verify reduced memory churn
