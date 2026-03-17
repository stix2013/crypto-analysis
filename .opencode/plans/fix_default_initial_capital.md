# Fix DEFAULT_INITIAL_CAPITAL Default Value Consistency

## Problem
In `worker/tasks.py`, line 18:
```python
DEFAULT_INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000"))
```

The default value is "10000" (integer string) but the original hardcoded value was 10000.0 (float). While float() conversion handles this correctly, using "10000.0" as the default string makes the intent clearer and ensures consistent type expectations.

## Solution
Change the default value from "10000" to "10000.0" in the os.getenv call to explicitly indicate we expect a float value.

## Changes Needed
In `worker/tasks.py`, line 18:
```python
DEFAULT_INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000.0"))
```

## Implementation Plan
This is a simple one-line change that improves code clarity without affecting functionality.

## Testing
After making the change, verify:
1. The default value is still correctly parsed as 10000.0
2. Environment variable override still works correctly (e.g., BACKTEST_INITIAL_CAPITAL=50000.5)
3. No existing functionality is broken
