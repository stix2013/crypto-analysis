# Developer Lessons Learned

This file tracks patterns, pitfalls, and corrections to prevent repeated mistakes in future sessions.

## Python & Machine Learning

### 1. Model Serialization (Pickle / Joblib)
- **Issue:** Using `defaultdict(lambda: ...)` inside a class prevents it from being serialized by `pickle` or `joblib`. This throws an `_pickle.PicklingError: Can't pickle local object`.
- **Solution:** Never use `lambda` functions (or local functions) for default initializers if the instance must be saved to disk. Instead, use a standard `dict` and implement a getter method that initializes missing keys, e.g., `if key not in self._data: self._data[key] = default_val`.

### 2. Scikit-Learn Deprecations
- **Issue:** `PassiveAggressiveClassifier` generates a `FutureWarning` in scikit-learn >= 1.8.
- **Solution:** For online classification via `partial_fit`, use `SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0)` as a direct, future-proof replacement.

## Third-Party APIs & Libraries

### 1. Binance Futures API
- **Issue:** Fetching Futures OHLCV (`fapi/v1/klines`) via official Python connectors can be overly complex or lack direct intuitive methods.
- **Solution:** Making direct HTTP requests via `requests.Session()` to `https://fapi.binance.com` is highly reliable and straightforward for public market data like klines/OHLCV. Always implement a backward-fetching loop with `startTime` / `endTime` bounds to gather large datasets (>1500 bars).

### 2. Mypy and External Libraries
- **Issue:** Untyped third-party packages (like `requests`) fail `mypy` checks with `[import-untyped]`.
- **Solution:** Append `# type: ignore[import-untyped]` to the import statements if installing type stubs (`types-requests`) is not currently an option. Always handle implicit `Any` returns from `.json()` calls explicitly via type ignores or casting.

## System Workflow

### 1. Model Inference Scripts
- **Pattern:** When designing prediction/inference scripts, do not just output `Signals: 0` if conditions aren't met.
- **Solution:** Always print detailed debug information (current regime, threshold, raw ensemble prediction values, component model predictions) so users understand *why* a signal wasn't generated. Models being conservative (e.g., ensemble score -0.175 < threshold 0.3) is normal behavior but looks like a bug without transparency.