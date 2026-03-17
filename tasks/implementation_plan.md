# Code Review & Fixes for `feature/visualize-results` Branch

Review of 20 changed files (1,230 additions, 104 deletions) against `main`. The branch adds visualization support, forward-fetching in `BinanceClient`, settings centralization for backtest config, and worker task updates.

## Issues Found

### 🔴 Critical: Code Will Crash at Runtime

---

#### Issue 1 — Missing `Backtester` methods and constructor params

[generate_backtest_plots.py](file:///home/stevan/projects/AI/trader/crypto-analysis/scripts/generate_backtest_plots.py) calls three things that **don't exist** on `Backtester`:

| Call | Line(s) | Status |
|------|---------|--------|
| `Backtester(generate_plots=True, output_dir=...)` | 137-142, 163-168 | `__init__` accepts `**kwargs` but ignores these |
| `backtester.set_price_data(price_data)` | 143, 169 | Method doesn't exist |
| `backtester._generate_visualization(history_df)` | 202 | Method doesn't exist |

[worker/tasks.py](file:///home/stevan/projects/AI/trader/crypto-analysis/worker/tasks.py) has the same problem — passes `generate_plots` to `Backtester` (line 291) and calls `set_price_data` (line 292).

**Fix**: Add `set_price_data`, `_generate_visualization`, and handle `generate_plots`/`output_dir` kwargs in `Backtester.__init__`.

---

#### Issue 2 — Entire code block duplicated in `generate_backtest_plots.py`

Lines 145-169 are an **exact duplicate** of lines 111-143. This means:
- `time_span_hours` and `bars_needed` are recomputed identically
- `Backtester` is instantiated **twice**
- `set_price_data` is called **twice**

The first backtester instance (lines 137-143) is immediately discarded when the second one is created (lines 163-169).

**Fix**: Remove the duplicated block (lines 145-169).

### 🟡 Minor Issues

---

#### Issue 3 — Unused `import os` in `worker/tasks.py`

`import os` was added at line 1 but is never used after settings centralization removed the `os.getenv()` calls.

**Fix**: Remove the unused import.

---

#### Issue 4 — Stale `.opencode/plans/` docs committed

Four plan files were added under `.opencode/plans/` but the AGENTS.md now says plans go in `tasks/`. These are historical artifacts from a previous session and shouldn't be committed.

**Fix**: Delete the `.opencode/plans/` files or move them to `tasks/`.

---

#### Issue 5 — `visualization.py` uses bare `except Exception`

[plot_feature_importance](file:///home/stevan/projects/AI/trader/crypto-analysis/src/crypto_analysis/visualization.py#L223) catches `Exception` broadly (line 223). Per project rules (`E722` is ignored but `SIM105` is explicitly preferred explicit), this should at minimum log, and ideally catch more specific exceptions.

**Fix**: Replace with `except (AttributeError, IndexError, ValueError)` to match likely failures.

---

#### Issue 6 — AGENTS.md has a typo: "Fizing"

Section header "Autonomous Bug Fizing" → should be "Autonomous Bug Fixing".

**Fix**: Correct the typo.

---

#### Issue 7 — `matplotlib` import in `analytics.py` moved to module-level risk

The `try/except ImportError` guard for matplotlib was removed from `plot_equity_curve`. Since `matplotlib` is now a hard dependency in `pyproject.toml`, this is fine — but the import is still **inside** the method body (lazy import). It should either stay lazy (for consistency) or move to module-level.

**Fix**: Move the import to module-level since it's now a required dependency.

## Proposed Changes

### Backtester (`backtest.py`)

#### [MODIFY] [backtest.py](file:///home/stevan/projects/AI/trader/crypto-analysis/src/crypto_analysis/signals/backtest.py)

- Add `generate_plots` and `output_dir` handling in `__init__` via `**kwargs`
- Add `set_price_data(data: pd.DataFrame)` method to pre-load price data
- Add `_generate_visualization(equity_df: pd.DataFrame)` method that uses `PerformanceAnalyzer.plot_equity_curve` and saves to `output_dir`

---

### Backtest Plots Script

#### [MODIFY] [generate_backtest_plots.py](file:///home/stevan/projects/AI/trader/crypto-analysis/scripts/generate_backtest_plots.py)

- Remove duplicated code block (lines 145-169)

---

### Worker Tasks

#### [MODIFY] [tasks.py](file:///home/stevan/projects/AI/trader/crypto-analysis/worker/tasks.py)

- Remove unused `import os`

---

### Analytics

#### [MODIFY] [analytics.py](file:///home/stevan/projects/AI/trader/crypto-analysis/src/crypto_analysis/utils/analytics.py)

- Move `import matplotlib.pyplot as plt` to module-level

---

### Visualization

#### [MODIFY] [visualization.py](file:///home/stevan/projects/AI/trader/crypto-analysis/src/crypto_analysis/visualization.py)

- Narrow `except Exception` to `except (AttributeError, IndexError, ValueError)`

---

### AGENTS.md

#### [MODIFY] [AGENTS.md](file:///home/stevan/projects/AI/trader/crypto-analysis/AGENTS.md)

- Fix "Fizing" → "Fixing" typo

---

### Stale Plans Cleanup

#### [DELETE] `.opencode/plans/fix_binance_fetch_historical.md`
#### [DELETE] `.opencode/plans/fix_default_initial_capital.md`
#### [DELETE] `.opencode/plans/improve_backtester_price_data_storage.md`
#### [DELETE] `.opencode/plans/update_fetch_historical_docstring.md`

These are stale plan docs from a previous session. Plans now belong in `tasks/`.

## Verification Plan

### Automated Tests

```bash
# 1. Run existing backtest tests — validates set_price_data doesn't break anything
pytest tests/signals/test_backtest.py -v

# 2. Run visualization tests
pytest tests/utils/test_visualization.py -v

# 3. Run analytics tests (validates matplotlib import move)
pytest tests/utils/test_analytics.py -v

# 4. Full test suite
pytest -v

# 5. Lint check
ruff check src/ scripts/ worker/
```

### New Tests Needed

- Add a test in `tests/signals/test_backtest.py` for `set_price_data` and `_generate_visualization` methods
- Verify that `Backtester(generate_plots=True, output_dir="/tmp")` doesn't crash

### Manual Verification

- Run `python scripts/generate_backtest_plots.py --help` to verify the script loads without import errors
