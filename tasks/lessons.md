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

## System Engineering & Backtesting

### 1. Data Engineering & Robustness
- **Issue:** Components requiring long lookbacks (e.g., `ma_200` or `StandardScaler`) cause `ValueError` or `KeyError` when passed insufficient data during tests or early-stream processing.
- **Solution:** Implement strict guard clauses checking `len(data)` against a `min_required` threshold before processing. Provide safe defaults (e.g., 1.0 for volatility multipliers) when data is insufficient to prevent pipeline crashes.

### 2. Trading Logic & Accounting
- **Issue:** Calculating Total Equity with short positions is error-prone when using "unrealized PnL" formulas if the `cash` balance already includes short-sale proceeds.
- **Solution:** Use the **Accounting Equation** approach for consistency: `Total Equity = Cash + Σ(Market Value of Longs) - Σ(Market Value of Shorts)`. This is easier to audit and avoids double-counting margin or proceeds.

### 3. Performance Testing (TDD)
- **Issue:** Integration tests for online learning pipelines can be extremely slow or timeout if they trigger full model retraining during simple buffer-limit checks.
- **Solution:** Use `mocker.patch.object` to disable expensive background tasks (like retraining) in unit tests that are only meant to verify data flow or buffer logic.

### 4. Backtesting Analytics
- **Issue:** Performance metrics (Sharpe/Sortino) produce `NaN` or `Inf` errors if a backtest period has zero trades or constant equity.
- **Solution:** Add `1e-10` epsilon to denominators and verify `len(returns) > 0` before calculating annualized ratios in the `PerformanceAnalyzer`.

### 5. Regime Detection Sensitivity
- **Issue:** Market regime detectors (Trending vs. Volatile) often fail in tests because synthetic data generators produce "volatility" that doesn't cross hardcoded standard deviation thresholds.
- **Solution:** Ensure test data generators use a high enough multiplier (e.g., `1.5` instead of `0.05`) for volatile regimes to ensure detection logic is actually exercised.
