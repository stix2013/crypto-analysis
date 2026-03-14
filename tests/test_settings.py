"""Tests for centralized settings module."""

import pytest
from crypto_analysis.settings import (
    BinanceSettings,
    RegimeSettings,
    Settings,
    get_settings,
)


class TestBinanceSettings:
    def test_defaults(self):
        settings = BinanceSettings()
        assert settings.api_key == ""
        assert settings.secret_key == ""
        assert settings.base_url == "https://fapi.binance.com"
        assert settings.testnet is False

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_SECRET_KEY", "test_secret")
        settings = BinanceSettings()
        assert settings.api_key == "test_key"
        assert settings.secret_key == "test_secret"

    def test_frozen(self):
        from pydantic import ValidationError

        settings = BinanceSettings(api_key="key")
        with pytest.raises(ValidationError):
            settings.api_key = "new_key"


class TestRegimeSettings:
    def test_defaults(self):
        settings = RegimeSettings()
        assert settings.threshold_base == 0.10
        assert settings.adjust_crash == 0.30

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("REGIME_THRESHOLD_BASE", "0.25")
        settings = RegimeSettings()
        assert settings.threshold_base == 0.25


class TestGetSettings:
    def test_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_manual_instantiation(self):
        # Even with singleton, we can create instances for tests
        custom_settings = Settings(binance=BinanceSettings(api_key="custom_key"))
        assert custom_settings.binance.api_key == "custom_key"
