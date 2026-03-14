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
        """Test loading from environment variables via Settings."""
        from crypto_analysis.settings import Settings

        monkeypatch.setenv("BINANCE_API_KEY", "env_api_key")
        monkeypatch.setenv("BINANCE_SECRET_KEY", "env_secret_key")

        # Mock get_settings to return a fresh Settings instance that reads the monkeypatched env
        with patch("crypto_analysis.data.binance.get_settings") as mock_get:
            mock_get.return_value = Settings()
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

    def test_request_signed(self, client):
        """Test signed request includes signature and timestamp."""
        config = BinanceConfig(api_key="test_key", secret_key="test_secret")
        client = BinanceClient(config)

        with patch.object(client, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = []
            mock_response.raise_for_status = MagicMock()
            mock_session.get.return_value = mock_response

            client._request(
                "GET", "/api/v3/account", params={"foo": "bar"}, signed=True
            )

            mock_session.get.assert_called_once()
            call_args = mock_session.get.call_args
            params = call_args.kwargs.get("params", {})

            assert "timestamp" in params
            assert "signature" in params

    def test_request_post_method(self, client):
        """Test POST request method."""
        with patch.object(client, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response

            result = client._request(
                "POST", "/api/v3/order", params={"symbol": "BTCUSDT"}
            )

            mock_session.post.assert_called_once()
            assert result == {"success": True}

    def test_request_unsupported_method(self, client):
        """Test unsupported HTTP method raises error."""
        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            client._request("DELETE", "/api/v3/test")

    def test_request_http_error(self, client):
        """Test HTTP error handling."""
        import requests

        with patch.object(client, "session") as mock_session:
            mock_response = MagicMock()
            error = requests.HTTPError("500 Server Error")
            mock_response.raise_for_status.side_effect = error
            mock_session.get.return_value = mock_response

            with pytest.raises(requests.HTTPError):
                client._request("GET", "/api/v3/test")

    def test_fetch_ohlcv_with_start_time(self, client):
        """Test fetching OHLCV with start time."""
        with patch.object(client, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = [
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
            mock_response.raise_for_status = MagicMock()
            mock_session.get.return_value = mock_response

            df = client.fetch_ohlcv("BTCUSDT", "1h", start_str="2023-01-01")

            assert len(df) == 1
            mock_session.get.assert_called_once()
            call_args = mock_session.get.call_args
            params = call_args.kwargs.get("params", {})
            assert "startTime" in params

    def test_fetch_ohlcv_with_end_time(self, client):
        """Test fetching OHLCV with end time."""
        with patch.object(client, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = [
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
            mock_response.raise_for_status = MagicMock()
            mock_session.get.return_value = mock_response

            df = client.fetch_ohlcv("BTCUSDT", "1h", end_str="2023-12-31")

            assert len(df) == 1
            mock_session.get.assert_called_once()
            call_args = mock_session.get.call_args
            params = call_args.kwargs.get("params", {})
            assert "endTime" in params

    def test_parse_time_iso_string(self):
        """Test time parsing with ISO string."""
        client = BinanceClient()
        result = client._parse_time("2023-01-01T00:00:00Z")
        assert result is not None
        assert isinstance(result, int)

    def test_parse_time_invalid(self):
        """Test time parsing with invalid input."""
        client = BinanceClient()
        with pytest.raises(ValueError, match="Invalid time format"):
            client._parse_time(None)
