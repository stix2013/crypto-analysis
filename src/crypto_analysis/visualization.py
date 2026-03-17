"""Visualization utilities for training and backtesting results."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from crypto_analysis.signals.base import SignalType


def plot_price_with_trades(
    data: pd.DataFrame,
    signals: pd.DataFrame | None = None,
    symbol: str = "BTCUSDT",
    title: str | None = None,
) -> Figure:
    """Plot OHLC price with buy/sell signal markers.

    Args:
        data: DataFrame with at least 'close' column and DatetimeIndex
        signals: DataFrame with 'timestamp', 'signal_type', and optionally 'price'
        symbol: Trading symbol name
        title: Custom plot title

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(data.index, data["close"], label="Close Price", color="gray", alpha=0.6)

    if signals is not None and not signals.empty:
        # Filter for entry signals
        buys = signals[signals["signal_type"] == SignalType.ENTRY_LONG.name]
        sells = signals[signals["signal_type"] == SignalType.ENTRY_SHORT.name]
        exits = signals[
            signals["signal_type"].isin(
                [SignalType.EXIT_LONG.name, SignalType.EXIT_SHORT.name]
            )
        ]

        # If signals don't have price, use close price from data
        if "price" not in signals.columns:
            buys = buys.merge(data[["close"]], left_on="timestamp", right_index=True)
            sells = sells.merge(data[["close"]], left_on="timestamp", right_index=True)
            exits = exits.merge(data[["close"]], left_on="timestamp", right_index=True)
            buy_prices = buys["close"]
            sell_prices = sells["close"]
            exit_prices = exits["close"]
        else:
            buy_prices = buys["price"]
            sell_prices = sells["price"]
            exit_prices = exits["price"]

        ax.scatter(
            buys["timestamp"],
            buy_prices,
            marker="^",
            color="green",
            s=100,
            label="Buy Signal",
            zorder=5,
        )
        ax.scatter(
            sells["timestamp"],
            sell_prices,
            marker="v",
            color="red",
            s=100,
            label="Sell Signal",
            zorder=5,
        )
        ax.scatter(
            exits["timestamp"],
            exit_prices,
            marker="x",
            color="black",
            s=50,
            label="Exit Signal",
            zorder=4,
        )

    ax.set_title(title or f"{symbol} Price and Trading Signals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_signal_distribution(
    signals: pd.DataFrame, title: str = "Signal Distribution"
) -> Figure:
    """Plot distribution of generated signals.

    Args:
        signals: DataFrame with 'signal_type' column
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = signals["signal_type"].value_counts()
    colors = []
    for label in counts.index:
        if "LONG" in label or "BUY" in label:
            colors.append("green")
        elif "SHORT" in label or "SELL" in label:
            colors.append("red")
        else:
            colors.append("gray")

    counts.plot(kind="bar", ax=ax, color=colors, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Signal Type")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_regime_timeline(
    signals: pd.DataFrame, title: str = "Market Regime Timeline"
) -> Figure:
    """Plot market regime changes over time.

    Args:
        signals: DataFrame with 'timestamp' and 'regime' columns
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    if "regime" not in signals.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No regime data available", ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(15, 4))

    # Convert regimes to numeric for plotting
    unique_regimes = sorted(signals["regime"].unique())
    regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
    y_values = signals["regime"].map(regime_map)

    ax.step(signals["timestamp"], y_values, where="post", color="blue", alpha=0.6)
    ax.set_yticks(range(len(unique_regimes)))
    ax.set_yticklabels(unique_regimes)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    generator: Any, title: str = "Feature Importance"
) -> Figure:
    """Plot top features by importance from the online model.

    Args:
        generator: OnlineSignalGenerator instance
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    try:
        # Handle OnlineRandomForest from the generator
        # generator.ml_model is likely an OnlineRandomForest or ensemble
        model = getattr(generator, "ml_model", None)
        feature_names = generator.feature_engineer.get_feature_columns(
            pd.DataFrame(columns=generator.feature_engineer.feature_names)
        )

        # Try to get importances from RF trees
        if model is not None and hasattr(model, "trees") and model.trees:
            # Average importance across all trained trees
            importances = np.zeros(len(feature_names))
            count = 0
            for tree in model.trees:
                if tree is not None:
                    importances += tree.feature_importances_
                    count += 1

            if count > 0:
                importances /= count

                # Sort and plot top 20
                indices = np.argsort(importances)[-20:]
                ax.barh(range(len(indices)), importances[indices], align="center")
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices])
                ax.set_xlabel("Relative Importance")
                ax.set_title(title)
            else:
                ax.text(0.5, 0.5, "Model trees not yet trained", ha="center")
        else:
            ax.text(
                0.5,
                0.5,
                "Feature importance only available for Random Forest",
                ha="center",
            )

    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting importance: {str(e)}", ha="center")

    plt.tight_layout()
    return fig


def save_training_plots(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    generator: Any,
    output_dir: str,
    prefix: str = "btc_1h",
) -> list[str]:
    """Generate and save all training plots.

    Args:
        data: Historical data used
        signals: Generated signals
        generator: Trained signal generator
        output_dir: Directory to save plots
        prefix: Filename prefix

    Returns:
        List of paths to saved plot files
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # 1. Price + Trades
    fig = plot_price_with_trades(data, signals, title=f"Training Results: {prefix}")
    path = os.path.join(output_dir, f"{prefix}_price.png")
    fig.savefig(path)
    plt.close(fig)
    saved_paths.append(path)

    # 2. Signal Distribution
    fig = plot_signal_distribution(signals)
    path = os.path.join(output_dir, f"{prefix}_signals.png")
    fig.savefig(path)
    plt.close(fig)
    saved_paths.append(path)

    # 3. Regime Timeline
    fig = plot_regime_timeline(signals)
    path = os.path.join(output_dir, f"{prefix}_regimes.png")
    fig.savefig(path)
    plt.close(fig)
    saved_paths.append(path)

    # 4. Feature Importance
    fig = plot_feature_importance(generator)
    path = os.path.join(output_dir, f"{prefix}_features.png")
    fig.savefig(path)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths
