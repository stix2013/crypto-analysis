"""Basic usage example for crypto analysis signals."""

import warnings

import numpy as np
import pandas as pd

from crypto_analysis.signals import (
    FeatureEngineer,
    LSTMSignalGenerator,
    RandomForestSignalGenerator,
    SignalAggregator,
    StatisticalArbitrageGenerator,
    TechnicalPatternGenerator,
)


def create_sample_data(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    freq: str = "1h",
) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration.

    Args:
        start_date: Start date
        end_date: End date
        freq: Data frequency

    Returns:
        OHLCV DataFrame

    """
    np.random.seed(42)
    dates = pd.date_range(start_date, end_date, freq=freq)
    returns = np.random.normal(0.0001, 0.02, len(dates))
    trend = np.sin(np.linspace(0, 8 * np.pi, len(dates))) * 0.001
    price = 30000 * np.exp(np.cumsum(returns + trend))

    df = pd.DataFrame(
        {
            "open": price * (1 + np.random.normal(0, 0.001, len(dates))),
            "high": price * (1 + abs(np.random.normal(0, 0.01, len(dates)))),
            "low": price * (1 - abs(np.random.normal(0, 0.01, len(dates)))),
            "close": price,
            "volume": np.random.uniform(100, 1000, len(dates)) * price,
        },
        index=dates,
    )

    return df


def example_usage():
    """Demonstrate complete signal generation pipeline."""
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data()

    # Split train/test
    train_df = df[: int(len(df) * 0.7)]
    test_df = df[int(len(df) * 0.7) :]

    print(f"Training data: {len(train_df)} bars")
    print(f"Test data: {len(test_df)} bars")

    # Create signal generators
    print("\nCreating and training signal generators...")

    # 1. LSTM Generator (reduced epochs for demo)
    print("\n1. Training LSTM...")
    LSTMSignalGenerator(
        sequence_length=60,
        lstm_units=[64, 32],
        dropout=0.2,
    )
    # Note: LSTM training requires significant data and time
    # Uncomment to actually train:
    # lstm_gen.fit(train_df, epochs=10)

    # 2. Random Forest Generator
    print("\n2. Training Random Forest...")
    rf_gen = RandomForestSignalGenerator(n_estimators=50)  # Reduced for speed
    rf_gen.fit(train_df)

    # 3. Technical Patterns (no training needed)
    print("\n3. Technical Pattern Generator...")
    tech_gen = TechnicalPatternGenerator()
    tech_gen.fit(train_df)

    # 4. Statistical Arbitrage (no training needed)
    print("\n4. Statistical Arbitrage Generator...")
    stat_gen = StatisticalArbitrageGenerator()
    stat_gen.fit(train_df)

    # Create aggregator
    print("\nCreating signal aggregator...")
    aggregator = SignalAggregator(method="weighted_confidence")
    # aggregator.add_generator(lstm_gen, weight=1.5)  # Uncomment when trained
    aggregator.add_generator(rf_gen, weight=1.0)
    aggregator.add_generator(tech_gen, weight=0.8)
    aggregator.add_generator(stat_gen, weight=1.0)

    # Test signal generation
    print("\nGenerating signals on test data...")
    signals_generated = 0

    for i in range(100, len(test_df), 24):  # Check every 24 hours
        window = test_df.iloc[:i]

        all_signals = []
        for gen in aggregator.generators:
            try:
                sigs = gen.generate(window)
                all_signals.extend(sigs)
            except Exception as e:
                warnings.warn(f"Generator {gen.name} failed: {e}", stacklevel=2)

        if all_signals:
            final = aggregator.aggregate(all_signals)
            if final:
                signals_generated += 1
                print(
                    f"{window.index[-1]}: {final.signal_type.value} "
                    f"(confidence: {final.confidence:.2f}) from {final.source}"
                )

    print(f"\nTotal signals generated: {signals_generated}")

    # Feature engineering example
    print("\n\nFeature Engineering Example:")
    fe = FeatureEngineer()
    features = fe.create_features(df.head(200))
    feature_cols = fe.get_feature_columns(features)
    print(f"Generated {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols[:10]):
        print(f"  - {col}")
    print(f"  ... and {len(feature_cols) - 10} more")

    return aggregator


if __name__ == "__main__":
    print("=" * 60)
    print("Crypto Analysis - Signal Generation Example")
    print("=" * 60)
    aggregator = example_usage()
    print("\nDone!")
