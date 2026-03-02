---
name: market-regime-expert
description: Adaptive strategy and risk management based on market regime detection. Use when you want to optimize trading performance by automatically adjusting bot behavior (sizing, stop-loss, risk-off) in different market conditions like trending, ranging, or volatile.
---

# Market Regime Expert

This skill provides the logic and templates needed to make your trading bot "market aware."

## Core Workflow

1.  **Map your strategy**: Review the [Regime Mapping](references/regime_mapping.md) to decide how your bot should react to each market state.
2.  **Use the Template**: Use `assets/adaptive_strategy_template.py` as a starting point for your new `AdaptiveRegimeStrategy`.
3.  **Analyze results**: Use `scripts/regime_backtest_analyzer.py` after a backtest to see which regimes are your most and least profitable.

## Adaptive Strategy Guide

### Quick Start

```python
from crypto_analysis.online.detection.regime import RegimeDetector
from your_module import AdaptiveRegimeStrategy

# Initialize with regime detection
regime_detector = RegimeDetector()
strategy = AdaptiveRegimeStrategy(symbols=["BTCUSDT"], aggregator=aggregator, regime_detector=regime_detector)
```

### Strategy Adjustments

| Regime | Adjustment Strategy |
| :--- | :--- |
| **Trending Up** | Maximize size, use wide trailing stop. |
| **Ranging** | Reduce size, use mean-reversion signals. |
| **Volatile** | Reduce size by 50%, tighten stop-loss. |
| **Crash** | **Risk Off**: Close all positions immediately. |

### Analysis

Run the analysis script to see if your adjustments are working:
`python scripts/regime_backtest_analyzer.py`
