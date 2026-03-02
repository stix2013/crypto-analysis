"""Tests for Binance data fetching module."""

from unittest.mock import MagicMock, patch

import pytest

from crypto_analysis.data.binance import BinanceClient, BinanceConfig, create_client


class TestBinanceConfig:
    """Tests for BinanceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration loads from environment."""
        config = BinanceConfig()
        assert config.base_url == "https://fapi.binance.com"
        assert config.testnet is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = BinanceConfig(api_key="test_key", secret_key="test_secret")
        assert config.api_key == "test_key"
        assert config.secret_key == "test_secret"

    def test_env_vars(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("BINANCE_API_KEY", "env_api_key")
        monkeypatch.setenv("BINANCE_SECRET_KEY", "env_secret_key")
        config = BinanceConfig()
        assert config.api_key == "env_api_key"
        assert config.secret_key == "env_secret_key"


class TestBinanceClient:
    """Tests for BinanceClient."""

    @pytest.fixture
    def mock_response(self):
        """Create mock response for testing."""
        mock = MagicMock()
        mock.json.return_value = [
            [
                1700000000000,
                "1900.0",
                "1910.0",
                "1890.0",
                "1905.0",
                "1000.0",
                1700000060000,
                "1900000.0",
                100,
                "500.0",
                "950000.0",
                "0",
            ]
        ]
        mock.raise_for_status = MagicMock()
        return mock

    def test_create_client(self):
        """Test client creation."""
        client = create_client()
        assert isinstance(client, BinanceClient)

    def test_parse_time_string(self):
        """Test time parsing with string."""
        client = BinanceClient()
        result = client._parse_time("1700000000000")
        assert result == 1700000000000

    def test_parse_time_int(self):
        """Test time parsing with integer."""
        client = BinanceClient()
        result = client._parse_time(1700000000000)
        assert result == 1700000000000

    def test_generate_signature(self):
        """Test HMAC signature generation."""
        config = BinanceConfig(api_key="key", secret_key="secret")
        client = BinanceClient(config)
        signature = client._generate_signature("symbol=ETHUSDT&timestamp=123")
        assert isinstance(signature, str)
        assert len(signature) == 64


class TestBinanceOHLCV:
    """Tests for OHLCV data fetching."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return BinanceClient()

    def test_fetch_ohlcv_structure(self, client):
        """Test that fetch_ohlcv returns correct DataFrame structure."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = [
                [
                    1700000000000,
                    "1900.0",
                    "1910.0",
                    "1890.0",
                    "1905.0",
                    "1000.0",
                    1700000060000,
                    "1900000.0",
                    100,
                    "500.0",
                    "950000.0",
                    "0",
                ]
            ]
            df = client.fetch_ohlcv("ETHUSDT", "15m", limit=1)

            assert len(df) == 1
            assert "open" in df.columns
            assert "high" in df.columns
            assert "low" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns

    def test_fetch_historical(self, client):
        """Test historical data fetching."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = [
                [
                    1700000000000,
                    "1900.0",
                    "1910.0",
                    "1890.0",
                    "1905.0",
                    "1000.0",
                    1700000060000,
                    "1900000.0",
                    100,
                    "500.0",
                    "950000.0",
                    "0",
                ]
            ]
            df = client.fetch_historical("ETHUSDT", "15m", bars=1)
            assert len(df) == 1
