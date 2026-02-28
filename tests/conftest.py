"""Test fixtures and utilities."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="1h")
    price = 30000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, 500)))

    df = pd.DataFrame(
        {
            "open": price * (1 + np.random.normal(0, 0.001, 500)),
            "high": price * (1 + abs(np.random.normal(0, 0.01, 500))),
            "low": price * (1 - abs(np.random.normal(0, 0.01, 500))),
            "close": price,
            "volume": np.random.uniform(100, 1000, 500) * price,
        },
        index=dates,
    )

    return df


@pytest.fixture
def minimal_ohlcv_data():
    """Create minimal OHLCV data for quick tests."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    price = np.linspace(30000, 31000, 100)

    df = pd.DataFrame(
        {
            "open": price * 0.999,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": np.ones(100) * 1000,
        },
        index=dates,
    )

    return df
