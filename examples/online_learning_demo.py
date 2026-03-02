"""Demo of online learning capabilities for crypto trading.

This script demonstrates:
- Real-time regime detection and adaptation
- Ensemble of online learning models (LSTM, NN, RF, PA)
- Adaptive learning rate based on market conditions
- Continuous model improvement with A/B testing
"""

import numpy as np
import pandas as pd

from crypto_analysis.online.generator import OnlineSignalGenerator
from crypto_analysis.online.pipeline import ContinuousLearningPipeline


def generate_synthetic_data(n_points: int = 5000) -> pd.DataFrame:
    """Generate synthetic market data with regime changes.

    Args:
        n_points: Number of data points to generate

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    regimes = [
        ("trending_up", 500, 0.0005, 0.01),
        ("volatile", 500, 0.0, 0.04),
        ("trending_down", 500, -0.0003, 0.015),
        ("ranging", 500, 0.0, 0.008),
        ("crash", 500, -0.001, 0.05),
    ]

    prices = [30000]
    for _regime, length, drift, vol in regimes:
        for _ in range(length):
            ret = np.random.normal(drift, vol)
            prices.append(prices[-1] * (1 + ret))

    dates = pd.date_range("2023-01-01", periods=len(prices), freq="1h")
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            "close": prices,
            "volume": np.random.uniform(100, 1000, len(prices)),
        },
        index=dates,
    )

    return df


def demo_online_learning() -> None:
    """Demonstrate online learning capabilities."""
    print("=" * 60)
    print("GENERATING SYNTHETIC MARKET DATA")
    print("=" * 60)

    df = generate_synthetic_data()

    train_df = df.iloc[:1000]
    stream_df = df.iloc[1000:]

    print("=" * 60)
    print("ONLINE LEARNING DEMO")
    print("=" * 60)

    generator = OnlineSignalGenerator(sequence_length=60)
    generator.fit(train_df)

    print("\nSimulating market data stream...")
    signals_generated = 0
    regime_changes_detected = 0

    for i in range(100, len(stream_df), 24):
        window = stream_df.iloc[:i]

        signals = generator.generate(window)

        if signals:
            signals_generated += 1
            sig = signals[0]
            print(
                f"\n{window.index[-1]} | "
                f"Regime: {generator.regime_detector.current_regime.name:12} | "
                f"Signal: {sig.signal_type.value:12} | "
                f"Confidence: {sig.confidence:.2f} | "
                f"LR: {generator.lr_scheduler.current_lr:.6f}"
            )

            weights = sig.metadata.get("model_weights", {})
            print(
                f"         Models: "
                f"LSTM={weights.get('lstm', 0):.2f} "
                f"NN={weights.get('nn', 0):.2f} "
                f"RF={weights.get('rf', 0):.2f} "
                f"PA={weights.get('pa', 0):.2f}"
            )

        if len(generator.regime_detector.regime_history) > 1 and (
            generator.regime_detector.regime_history[-1].regime_id
            != generator.regime_detector.regime_history[-2].regime_id
        ):
            regime_changes_detected += 1

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Total signals generated: {signals_generated}")
    print(f"  Regime changes detected: {regime_changes_detected}")
    print(f"  Final model weights: {generator.model_weights}")
    print(f"  Current learning rate: {generator.lr_scheduler.current_lr:.6f}")
    print(f"  Current regime: {generator.regime_detector.current_regime.name}")
    print(f"{'=' * 60}")


def demo_continuous_pipeline() -> None:
    """Demonstrate continuous learning pipeline."""
    print("\n" + "=" * 60)
    print("CONTINUOUS LEARNING PIPELINE DEMO")
    print("=" * 60)

    df = generate_synthetic_data(3000)

    pipeline = ContinuousLearningPipeline(checkpoint_dir="./models")

    for i in range(100, len(df), 50):
        window = df.iloc[:i]
        pipeline.stream_data(window)

        signals, source = pipeline.get_prediction(window)
        if signals:
            print(
                f"[{window.index[-1]}] Source: {source} - Signal: {signals[0].signal_type.value}"
            )

    print(f"\nA/B Testing Active: {pipeline.ab_test_active}")
    print(f"Models in buffer: {len(pipeline.data_buffer)}")


if __name__ == "__main__":
    demo_online_learning()
    demo_continuous_pipeline()
