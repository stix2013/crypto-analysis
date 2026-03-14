"""Simple prediction API for trained models.

Provides a clean interface for loading models and generating predictions
with automatic model path resolution based on symbol and interval.
"""

from pathlib import Path
from typing import Any, Optional, cast

import joblib
import pandas as pd

from crypto_analysis.data import Interval, create_client
from crypto_analysis.settings import get_settings
from crypto_analysis.signals.base import Signal


def resolve_model_path(
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    model_path: Optional[str] = None,
    models_dir: str = "models",
) -> Path:
    """Resolve model path from symbol+interval or explicit path.

    Resolution priority:
    1. Explicit model_path → use directly
    2. Symbol provided:
       - With interval → models/model_{symbol}_{interval}.joblib
       - Without interval → models/model_{symbol}_{PREDICT_INTERVAL}.joblib
    3. No symbol → use PREDICT_MODEL from environment

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        interval: Kline interval (e.g., 1h, 15m)
        model_path: Explicit model file path
        models_dir: Base directory for models

    Returns:
        Resolved model path

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If no model can be resolved

    Examples:
        >>> resolve_model_path("BTCUSDT", "1h")
        Path('models/model_btcusdt_1h.joblib')

        >>> resolve_model_path("BTCUSDT")  # Uses PREDICT_INTERVAL env
        Path('models/model_btcusdt_1h.joblib')
    """
    settings = get_settings()

    # Priority 1: Explicit model path
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    # Priority 2: Symbol provided
    if symbol:
        symbol_lower = symbol.lower()

        # Determine interval: use provided, then env, then default
        resolved_interval = interval or settings.predict.interval

        # Build filename: model_{symbol}_{interval}.joblib
        filename = f"model_{symbol_lower}_{resolved_interval}.joblib"
        path = Path(models_dir) / filename

        if not path.exists():
            # Try with model name from settings if it matches
            if (
                settings.predict.model
                and (Path(models_dir) / settings.predict.model).exists()
            ):
                # This is a bit complex, let's stick to the logic
                pass

            if not path.exists():
                raise FileNotFoundError(
                    f"Model not found: {path}\n"
                    f"Tried to resolve: symbol='{symbol}', interval='{resolved_interval}'\n"
                    f"Expected filename: {filename}"
                )
        return path

    # Priority 3: Use PREDICT_MODEL from environment
    env_model = settings.predict.model
    if env_model:
        path = Path(env_model)
        if not path.exists():
            # Try with models_dir prefix
            path = Path(models_dir) / env_model
            if not path.exists():
                raise FileNotFoundError(
                    f"Model from PREDICT_MODEL not found: {env_model}\n"
                    f"Tried: {Path(models_dir) / env_model}"
                )
        return path

    raise ValueError(
        "No model specified. Provide one of:\n"
        "  - model_path: Explicit path to model file\n"
        "  - symbol: Trading pair symbol (will use PREDICT_INTERVAL or '1h')\n"
        "  - PREDICT_MODEL environment variable"
    )


class Predictor:
    """Reusable predictor for efficient inference.

    Loads a model once and allows multiple predictions without reloading.
    Automatically fetches data from Binance if not provided.

    Attributes:
        model: Loaded signal generator model
        model_path: Path to the loaded model
        symbol: Trading symbol
        interval: Data interval

    Examples:
        >>> predictor = Predictor("BTCUSDT", interval="1h")
        >>> signals = predictor.predict()
        >>> signals = predictor.predict(bars=500)  # More bars
    """

    model: Any
    model_path: Path
    symbol: str
    interval: Interval

    def __init__(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        model_path: Optional[str] = None,
        models_dir: str = "models",
    ) -> None:
        """Initialize predictor with auto model resolution.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (used with symbol)
            model_path: Explicit model file path
            models_dir: Base directory for models

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If no model can be resolved
        """
        settings = get_settings()
        self.model_path = resolve_model_path(symbol, interval, model_path, models_dir)
        self.model = joblib.load(self.model_path)

        # Extract symbol from model or provided value
        self.symbol = symbol or settings.predict.symbol
        raw_interval = interval or settings.predict.interval
        self.interval = cast(Interval, raw_interval)

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        bars: int = 200,
    ) -> list[Signal]:
        """Generate predictions.

        Args:
            data: OHLCV DataFrame. If None, fetches from Binance.
            bars: Number of bars to fetch if data not provided

        Returns:
            List of trading signals

        Raises:
            ValueError: If insufficient data for prediction
        """
        # Fetch data if not provided
        if data is None:
            client = create_client()
            data = client.fetch_historical(self.symbol, self.interval, bars)

        # Add symbol info for the model
        if not hasattr(data, "symbol"):
            data = data.copy()
            if "symbol" not in data.columns:
                data.index.name = self.symbol

        # Generate signals
        signals = cast(list[Signal], self.model.generate(data))

        return signals


def predict(
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    model_path: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    bars: int = 200,
    models_dir: str = "models",
) -> list[Signal]:
    """Simple one-liner prediction.

    Automatically resolves model path, loads model, and generates signals.
    For repeated predictions, use Predictor class for better performance.

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        interval: Kline interval (e.g., 1h, 15m). Uses PREDICT_INTERVAL env if not provided.
        model_path: Explicit model file path (overrides symbol+interval)
        data: OHLCV DataFrame. If None, fetches from Binance.
        bars: Number of bars to fetch if data not provided
        models_dir: Base directory for models

    Returns:
        List of trading signals

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If no model can be resolved

    Examples:
        >>> signals = predict("BTCUSDT", interval="1h")
        >>> signals = predict(symbol="ETHUSDT")  # Uses PREDICT_INTERVAL
        >>> signals = predict(model_path="models/custom.joblib")
    """
    predictor = Predictor(symbol, interval, model_path, models_dir)
    return predictor.predict(data, bars)
