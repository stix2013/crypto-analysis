"""Tests for prediction API."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from crypto_analysis.signals.base import Signal, SignalType
from crypto_analysis.signals.predict import (
    Predictor,
    predict,
    resolve_model_path,
)


class TestResolveModelPath:
    """Test model path resolution logic."""

    def test_explicit_model_path(self, tmp_path: Path) -> None:
        """Test explicit model path takes priority."""
        model_file = tmp_path / "custom.joblib"
        model_file.touch()

        result = resolve_model_path(model_path=str(model_file))
        assert result == model_file

    def test_explicit_model_path_not_found(self) -> None:
        """Test error when explicit model path doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            resolve_model_path(model_path="nonexistent.joblib")

    def test_symbol_and_interval(self, tmp_path: Path) -> None:
        """Test resolution with symbol and interval."""
        model_file = tmp_path / "model_btcusdt_1h.joblib"
        model_file.touch()

        result = resolve_model_path(
            symbol="BTCUSDT", interval="1h", models_dir=str(tmp_path)
        )
        assert result == model_file

    def test_symbol_uses_predict_interval_env(self, tmp_path: Path) -> None:
        """Test symbol-only uses PREDICT_INTERVAL from settings."""
        from crypto_analysis.settings import Settings, PredictSettings

        model_file = tmp_path / "model_ethusdt_4h.joblib"
        model_file.touch()

        # Mock get_settings to return a Settings instance with the desired interval
        with patch("crypto_analysis.signals.predict.get_settings") as mock_get:
            mock_get.return_value = Settings(predict=PredictSettings(interval="4h"))
            result = resolve_model_path(symbol="ETHUSDT", models_dir=str(tmp_path))
            assert result == model_file

    def test_symbol_uses_default_interval_if_no_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test symbol-only uses default 1h when env not set."""
        monkeypatch.delenv("PREDICT_INTERVAL", raising=False)
        model_file = tmp_path / "model_btcusdt_1h.joblib"
        model_file.touch()

        result = resolve_model_path(symbol="BTCUSDT", models_dir=str(tmp_path))
        assert result == model_file

    def test_symbol_lowercase_in_filename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test symbol is lowercased in filename."""
        monkeypatch.delenv("PREDICT_INTERVAL", raising=False)
        model_file = tmp_path / "model_btcusdt_1h.joblib"
        model_file.touch()

        # Uppercase symbol should resolve to lowercase filename
        result = resolve_model_path(symbol="BTCUSDT", models_dir=str(tmp_path))
        assert result == model_file
        assert "btcusdt" in str(result)

    def test_predict_model_env(self, tmp_path: Path) -> None:
        """Test PREDICT_MODEL from settings."""
        from crypto_analysis.settings import Settings, PredictSettings

        model_file = tmp_path / "model_ethusdt_1h.joblib"
        model_file.touch()

        with patch("crypto_analysis.signals.predict.get_settings") as mock_get:
            mock_get.return_value = Settings(
                predict=PredictSettings(model="model_ethusdt_1h.joblib")
            )
            result = resolve_model_path(models_dir=str(tmp_path))
            assert result == model_file

    def test_predict_model_env_with_path(self, tmp_path: Path) -> None:
        """Test PREDICT_MODEL with full path from settings."""
        from crypto_analysis.settings import Settings, PredictSettings

        model_file = tmp_path / "custom_model.joblib"
        model_file.touch()

        with patch("crypto_analysis.signals.predict.get_settings") as mock_get:
            mock_get.return_value = Settings(
                predict=PredictSettings(model=str(model_file))
            )
            result = resolve_model_path(models_dir="/wrong/path")
            assert result == model_file

    def test_no_model_specified_error(self) -> None:
        """Test error when no model can be resolved."""
        from crypto_analysis.settings import Settings, PredictSettings

        with patch("crypto_analysis.signals.predict.get_settings") as mock_get:
            # Create settings with empty values
            mock_get.return_value = Settings(
                predict=PredictSettings(model="", interval="1h", symbol="")
            )
            with pytest.raises(ValueError, match="No model specified"):
                resolve_model_path()

    def test_symbol_model_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error when symbol-based model doesn't exist."""
        monkeypatch.delenv("PREDICT_INTERVAL", raising=False)

        with pytest.raises(FileNotFoundError, match="Model not found"):
            resolve_model_path(symbol="BTCUSDT", models_dir=str(tmp_path))

    def test_symbol_model_not_found_with_interval(self, tmp_path: Path) -> None:
        """Test error message includes correct filename pattern."""
        from crypto_analysis.settings import Settings, PredictSettings

        with patch("crypto_analysis.signals.predict.get_settings") as mock_get:
            mock_get.return_value = Settings(predict=PredictSettings(interval="1h"))
            with pytest.raises(FileNotFoundError) as exc_info:
                resolve_model_path(symbol="BTCUSDT", models_dir=str(tmp_path))

        assert "model_btcusdt_1h.joblib" in str(exc_info.value)


class TestPredictor:
    """Test Predictor class."""

    def test_init_with_symbol_and_interval(self) -> None:
        """Test Predictor initialization with symbol and interval."""
        mock_model = Mock()
        mock_model.name = "TestModel"
        mock_model.lookback_period = 100

        with patch(
            "crypto_analysis.signals.predict.joblib.load", return_value=mock_model
        ):
            with patch(
                "crypto_analysis.signals.predict.resolve_model_path",
                return_value=Path("models/model_btcusdt_1h.joblib"),
            ):
                predictor = Predictor(symbol="BTCUSDT", interval="1h")

                assert predictor.symbol == "BTCUSDT"
                assert predictor.interval == "1h"
                assert predictor.model == mock_model

    def test_init_with_explicit_model_path(self) -> None:
        """Test Predictor with explicit model path."""
        mock_model = Mock()
        mock_model.name = "TestModel"
        mock_model.lookback_period = 100

        with patch(
            "crypto_analysis.signals.predict.joblib.load", return_value=mock_model
        ):
            with patch(
                "crypto_analysis.signals.predict.resolve_model_path",
                return_value=Path("models/custom.joblib"),
            ):
                predictor = Predictor(model_path="models/custom.joblib")
                assert predictor.model == mock_model

    def test_predict_with_provided_data(self) -> None:
        """Test prediction with provided DataFrame."""
        mock_model = Mock()
        mock_model.lookback_period = 50
        mock_model.generate.return_value = [
            Signal(
                symbol="BTCUSDT",
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.8,
                timestamp=pd.Timestamp.now(),
                source="TestModel",
            )
        ]

        with patch(
            "crypto_analysis.signals.predict.joblib.load", return_value=mock_model
        ):
            with patch(
                "crypto_analysis.signals.predict.resolve_model_path",
                return_value=Path("models/model_btcusdt_1h.joblib"),
            ):
                predictor = Predictor(symbol="BTCUSDT", interval="1h")

                dates = pd.date_range("2024-01-01", periods=100, freq="1h")
                data = pd.DataFrame(
                    {
                        "open": range(100),
                        "high": range(100),
                        "low": range(100),
                        "close": range(100),
                        "volume": range(100),
                    },
                    index=dates,
                )

                signals = predictor.predict(data=data)
                assert len(signals) == 1
                assert signals[0].signal_type == SignalType.ENTRY_LONG
                mock_model.generate.assert_called_once()

    def test_predict_fetches_data_if_not_provided(self) -> None:
        """Test prediction auto-fetches data from Binance."""
        mock_model = Mock()
        mock_model.lookback_period = 50
        mock_model.generate.return_value = []

        with patch(
            "crypto_analysis.signals.predict.joblib.load", return_value=mock_model
        ):
            with patch(
                "crypto_analysis.signals.predict.resolve_model_path",
                return_value=Path("models/model_ethusdt_15m.joblib"),
            ):
                with patch(
                    "crypto_analysis.signals.predict.create_client"
                ) as mock_create:
                    mock_client = Mock()
                    mock_create.return_value = mock_client

                    dates = pd.date_range("2024-01-01", periods=100, freq="15min")
                    mock_data = pd.DataFrame(
                        {
                            "open": range(100),
                            "high": range(100),
                            "low": range(100),
                            "close": range(100),
                            "volume": range(100),
                        },
                        index=dates,
                    )
                    mock_client.fetch_historical.return_value = mock_data

                    predictor = Predictor(symbol="ETHUSDT", interval="15m")
                    signals = predictor.predict(bars=200)

                    mock_client.fetch_historical.assert_called_once_with(
                        "ETHUSDT", "15m", 200
                    )
                    mock_model.generate.assert_called_once()


class TestPredictFunction:
    """Test convenience predict() function."""

    def test_predict_one_liner(self) -> None:
        """Test simple predict() call."""
        mock_model = Mock()
        mock_model.lookback_period = 50
        mock_model.generate.return_value = []

        with patch(
            "crypto_analysis.signals.predict.joblib.load", return_value=mock_model
        ):
            with patch(
                "crypto_analysis.signals.predict.resolve_model_path",
                return_value=Path("models/model_btcusdt_1h.joblib"),
            ):
                with patch(
                    "crypto_analysis.signals.predict.create_client"
                ) as mock_create:
                    mock_client = Mock()
                    mock_create.return_value = mock_client

                    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
                    mock_data = pd.DataFrame(
                        {
                            "open": range(100),
                            "high": range(100),
                            "low": range(100),
                            "close": range(100),
                            "volume": range(100),
                        },
                        index=dates,
                    )
                    mock_client.fetch_historical.return_value = mock_data

                    signals = predict(symbol="BTCUSDT", interval="1h")

                    assert isinstance(signals, list)
                    mock_model.generate.assert_called_once()

    def test_predict_with_explicit_model_path(self) -> None:
        """Test predict() with explicit model path."""
        mock_model = Mock()
        mock_model.lookback_period = 50
        mock_model.generate.return_value = [
            Signal(
                symbol="BTCUSDT",
                signal_type=SignalType.ENTRY_SHORT,
                confidence=0.75,
                timestamp=pd.Timestamp.now(),
                source="TestModel",
            )
        ]

        with patch(
            "crypto_analysis.signals.predict.joblib.load", return_value=mock_model
        ):
            with patch(
                "crypto_analysis.signals.predict.resolve_model_path",
                return_value=Path("models/custom.joblib"),
            ):
                dates = pd.date_range("2024-01-01", periods=100, freq="1h")
                data = pd.DataFrame(
                    {
                        "open": range(100),
                        "high": range(100),
                        "low": range(100),
                        "close": range(100),
                        "volume": range(100),
                    },
                    index=dates,
                )

                signals = predict(model_path="models/custom.joblib", data=data)

                assert len(signals) == 1
                assert signals[0].signal_type == SignalType.ENTRY_SHORT


class TestIntegration:
    """Integration tests with real model."""

    def test_resolve_existing_model(self) -> None:
        """Test resolution of existing model in models/ directory."""
        # This test requires the model file to exist
        if not Path("models/model_ethusdt_15m.joblib").exists():
            pytest.skip("Model file not found")

        result = resolve_model_path(
            symbol="ETHUSDT",
            interval="15m",
            models_dir="models",
        )
        assert result == Path("models/model_ethusdt_15m.joblib")
