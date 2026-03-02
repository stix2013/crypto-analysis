# Market Regime Mapping

Use this guide to adjust `MLStrategy` parameters based on the detected market regime from `RegimeDetector`.

| Regime | Strategy Type | Risk Level | Stop-Loss | Take-Profit | Position Sizing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trending Up** | Trend Following | High | 2.5% (Wide) | 6% (Trailing) | 100% (Full) |
| **Trending Down** | Mean Reversion (Short) | Medium | 2% | 4% | 75% |
| **Ranging** | Mean Reversion (Both) | Low | 1.5% (Tight) | 2.5% | 50% |
| **Volatile** | Momentum Scalping | Very Low | 3% (Volatility Adj) | 4.5% | 25% |
| **Crash** | Risk Off | Zero | Close All | N/A | 0% (Cash Only) |

## Implementation Hints

- **Trending Up**: Loosen `stop_loss` to capture larger trends. Increase `target_size`.
- **Volatile**: Use `_volatility_adjustment` from `MLStrategy` but multiply by 0.5 to be more conservative.
- **Crash**: Immediately trigger `SignalType.RISK_OFF` to exit all positions.
- **Ranging**: Reduce `kelly_fraction` and use mean-reversion generators only.
