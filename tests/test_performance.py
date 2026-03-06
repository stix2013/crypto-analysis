import time

import numpy as np
import pandas as pd
from crypto_analysis.online.generator import OnlineSignalGenerator
from crypto_analysis.signals.features import FeatureEngineer


def test_feature_calculation_performance():
    """Benchmark the vectorized trend features against a large dataset."""
    fe = FeatureEngineer()

    # Create 10,000 rows of dummy data with POSITIVE values to avoid NaNs in indicators
    # Base price 100 + noise
    data = pd.DataFrame(
        {
            "open": 100 + np.random.randn(10000).cumsum(),
            "high": 105 + np.random.randn(10000).cumsum(),
            "low": 95 + np.random.randn(10000).cumsum(),
            "close": 100 + np.random.randn(10000).cumsum(),
            "volume": 1000 + np.abs(np.random.randn(10000) * 100),
        },
        index=pd.date_range("2020-01-01", periods=10000, freq="1h"),
    )

    start_time = time.time()
    features = fe.create_features(data)
    duration = time.time() - start_time

    print(f"\nFeature calculation for 10,000 rows took: {duration:.4f}s")

    # Check that trend features are present and calculated
    assert "trend_slope_14" in features.columns
    assert "trend_r2_14" in features.columns

    # Check that we actually have values (not all NaN)
    valid_values = features["trend_slope_14"].dropna()
    assert not valid_values.empty, "Trend slope contains only NaNs"

    # Vectorized calculation should be very fast
    assert duration < 2.0, f"Feature calculation is too slow: {duration:.4f}s"


def test_training_loop_complexity():
    """Verify that the training loop logic scales linearly."""

    # We test the core logic that was in train_model

    def simulate_training_loop(bars):
        # 1. Create data - enough for MA200 and other indicators
        total_bars = bars + 500
        data = pd.DataFrame(
            {
                "open": 100 + np.random.randn(total_bars).cumsum(),
                "high": 105 + np.random.randn(total_bars).cumsum(),
                "low": 95 + np.random.randn(total_bars).cumsum(),
                "close": 100 + np.random.randn(total_bars).cumsum(),
                "volume": 1000 + np.abs(np.random.randn(total_bars) * 100),
            },
            index=pd.date_range("2020-01-01", periods=total_bars, freq="1h"),
        )

        generator = OnlineSignalGenerator(
            name="Test",
            sequence_length=24,
            update_frequency=10,
        )

        # Warmup needs to be large enough for all features
        warmup_bars = 400
        warmup_data = data.iloc[:warmup_bars]
        generator.fit(warmup_data)

        start_time = time.time()

        # --- START OF LOGIC TO BENCHMARK ---
        # We simulate the loop over 'bars' number of points
        online_data = data.iloc[warmup_bars : warmup_bars + bars]

        # Optimized path: pre-calculate features
        all_features = generator.feature_engineer.create_features(data)

        for idx in online_data.index:
            lookback = data.loc[:idx]
            if len(lookback) < generator.lookback_period:
                continue

            current_features = all_features.loc[:idx]
            # Call generator with pre-calculated features
            _ = generator.generate(lookback, features_df=current_features)
        # --- END OF LOGIC TO BENCHMARK ---

        return time.time() - start_time

    # Measure for two different sizes
    size1 = 100
    size2 = 300

    time_small = simulate_training_loop(size1)
    time_large = simulate_training_loop(size2)

    ratio = time_large / (time_small or 0.001)

    print(f"\nTime for {size1} bars: {time_small:.4f}s")
    print(f"\nTime for {size2} bars: {time_large:.4f}s")
    print(f"\nScaling factor: {ratio:.2f}x time for {size2/size1:.2f}x data")

    # Linear scaling means ratio should be approx size2/size1 = 3.0
    # Quadratic scaling means ratio should be approx (size2/size1)^2 = 9.0
    # We assert ratio < 5.0 to be safe but clearly non-quadratic
    assert (
        ratio < 5.0
    ), f"Quadratic scaling detected! {size2}/{size1} data took {ratio:.2f}x longer."
