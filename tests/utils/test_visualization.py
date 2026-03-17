"""Tests for visualization utilities."""

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest
from crypto_analysis.signals.base import SignalType
from crypto_analysis.visualization import (
    plot_feature_importance,
    plot_price_with_trades,
    plot_regime_timeline,
    plot_signal_distribution,
    save_training_plots,
)
from matplotlib.figure import Figure


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    return pd.DataFrame(
        {
            "open": [100.0] * 100,
            "high": [110.0] * 100,
            "low": [90.0] * 100,
            "close": [105.0] * 100,
            "volume": [1000.0] * 100,
        },
        index=dates,
    )


@pytest.fixture
def sample_signals(sample_data):
    """Create sample signals."""
    timestamps = sample_data.index[10:20]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "signal_type": [
                SignalType.ENTRY_LONG.name,
                SignalType.EXIT_LONG.name,
                SignalType.ENTRY_SHORT.name,
                SignalType.EXIT_SHORT.name,
                SignalType.ENTRY_LONG.name,
                SignalType.ENTRY_LONG.name,
                SignalType.ENTRY_SHORT.name,
                SignalType.ENTRY_SHORT.name,
                SignalType.EXIT_LONG.name,
                SignalType.EXIT_SHORT.name,
            ],
            "regime": ["ranging"] * 5 + ["trending_up"] * 5,
            "confidence": [0.8] * 10,
        }
    )


def test_plot_price_with_trades(sample_data, sample_signals):
    """Test price plot with trades."""
    fig = plot_price_with_trades(sample_data, sample_signals)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_signal_distribution(sample_signals):
    """Test signal distribution plot."""
    fig = plot_signal_distribution(sample_signals)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_regime_timeline(sample_signals):
    """Test regime timeline plot."""
    fig = plot_regime_timeline(sample_signals)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_regime_timeline_no_data():
    """Test regime timeline with missing column."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp.now()]})
    fig = plot_regime_timeline(df)
    assert isinstance(fig, Figure)


def test_plot_feature_importance():
    """Test feature importance plot."""
    mock_generator = MagicMock()
    mock_model = MagicMock()
    mock_tree = MagicMock()
    mock_tree.feature_importances_ = [0.1, 0.2, 0.7]
    mock_model.trees = [mock_tree]
    mock_generator.ml_model = mock_model
    mock_generator.feature_engineer.feature_names = ["feat1", "feat2", "feat3"]
    mock_generator.feature_engineer.get_feature_columns.return_value = [
        "feat1",
        "feat2",
        "feat3",
    ]

    fig = plot_feature_importance(mock_generator)
    assert isinstance(fig, Figure)


def test_save_training_plots(sample_data, sample_signals, tmp_path):
    """Test saving all plots to a directory."""
    mock_generator = MagicMock()
    mock_generator.feature_engineer.feature_names = []
    mock_generator.feature_engineer.get_feature_columns.return_value = []

    output_dir = str(tmp_path / "plots")
    prefix = "test_run"

    paths = save_training_plots(
        sample_data, sample_signals, mock_generator, output_dir, prefix
    )

    assert len(paths) == 4
    for p in paths:
        assert os.path.exists(p)
        assert os.path.basename(p).startswith(prefix)
