# Problems Found and Fixed in Crypto Analysis Project

## 1. Inconsistent Behavior in BinanceClient.fetch_historical

**Problem**: The `fetch_historical` method behaved inconsistently depending on whether `start_time` was provided:
- When `start_time=None`: Used progressive fetching to get the full requested number of bars
- When `start_time` was provided: Used a single API request limited to 1500 bars max

**Impact**:
- `fetch_historical("BTCUSDT", "1h", 2000)` returned ~2000 bars
- `fetch_historical("BTCUSDT", "1h", 2000, start_time=some_timestamp)` returned max 1500 bars

**Solution**: Modified the method to use chunked fetching when `start_time` is provided, ensuring consistent behavior regardless of whether start_time is specified.

## 2. Inefficient Price Data Storage in Backtester._update_data_handler

**Problem**: The `_update_data_handler` method used `pd.concat` in a loop to build up price data:
```python
if timestamp not in self._price_data.index:
    new_row = pd.DataFrame({"close": [price]}, index=[timestamp])
    self._price_data = pd.concat([self._price_data, new_row])
```

**Impact**: O(N²) time complexity due to creating new DataFrame and copying all existing data on each update.

**Solution**: Replaced with a buffered approach that collects rows in a list and periodically concatenates in batches, reducing time complexity to O(N).

## 3. Inconsistent Default Value Handling for DEFAULT_INITIAL_CAPITAL

**Problem**: In `worker/tasks.py`, the default value was "10000" (integer string) while the original hardcoded value was 10000.0 (float).

**Impact**: While functionally equivalent due to float() conversion, it was unclear and inconsistent.

**Solution**: Changed default to "10000.0" for clarity, then later improved by accessing the value through the settings system.

## 4. Inaccurate Docstring for fetch_historical Method

**Problem**: The docstring misleadingly stated "default: fetch latest" without explaining that:
- When start_time=None: Uses backward-fetching approach to get most recent bars
- When start_time is provided: Fetches forward from the specified time

**Impact**: Developers could misunderstand the method's behavior.

**Solution**: Updated the docstring to accurately describe both behaviors.

## 5. Direct Environment Variable Access

**Problem**: Several locations accessed environment variables directly via `os.getenv()` instead of through the centralized settings system:
- `worker/tasks.py`: Direct access to `BACKTEST_INITIAL_CAPITAL` and `ENABLE_BACKTEST_PLOTS`
- `scripts/generate_backtest_plots.py`: Direct access to `BACKTEST_INITIAL_CAPITAL`

**Impact**: Violated project convention stated in AGENTS.md that all environment variables should be accessed through `src/crypto_analysis/settings.py`.

**Solution**:
- Created `BacktestSettings` class in `settings.py`
- Updated `worker/tasks.py` and `scripts/generate_backtest_plots.py` to access settings through `get_settings()`
- Updated documentation to reflect the proper approach

## Files Modified

1. `src/crypto_analysis/data/binance.py` - Fixed fetch_historical behavior and docstring
2. `src/crypto_analysis/signals/backtest.py` - Improved price data storage efficiency
3. `worker/tasks.py` - Removed direct os.getenv() calls, added settings access
4. `scripts/generate_backtest_plots.py` - Removed direct os.getenv() call
5. `src/crypto_analysis/settings.py` - Added BacktestSettings class
6. `README.md` - Added documentation about settings usage
7. `AGENTS.md` - Added environment variable access rule

All changes maintain backward compatibility while improving code consistency, performance, and adherence to project conventions.
