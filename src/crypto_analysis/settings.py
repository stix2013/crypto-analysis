"""Centralized settings management using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BinanceSettings(BaseSettings):
    """Binance API configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BINANCE_",
        frozen=True,
        extra="ignore",
    )

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://fapi.binance.com"
    testnet: bool = False


class RegimeSettings(BaseSettings):
    """Market regime threshold configuration."""

    model_config = SettingsConfigDict(
        env_prefix="REGIME_",
        frozen=True,
        extra="ignore",
    )

    threshold_base: float = 0.10
    adjust_trending_up: float = 0.05
    adjust_trending_down: float = 0.05
    adjust_ranging: float = 0.20
    adjust_volatile: float = 0.15
    adjust_crash: float = 0.30


class TrainSettings(BaseSettings):
    """Training script configuration."""

    model_config = SettingsConfigDict(
        env_prefix="TRAIN_",
        frozen=True,
        extra="ignore",
    )

    symbol: str = "ETHUSDT"
    interval: str = "15m"
    bars: int = 5000
    warmup_bars: int = 1000
    sequence_length: int = 60


class PredictSettings(BaseSettings):
    """Prediction script configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PREDICT_",
        frozen=True,
        extra="ignore",
    )

    model: str = "model_1h_ethusdt.joblib"
    symbol: str = "ETHUSDT"
    interval: str = "1h"
    bars: int = 300


class CelerySettings(BaseSettings):
    """Celery worker configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CELERY_",
        frozen=True,
        extra="ignore",
    )

    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    log_level: str = "info"
    worker_concurrency: int = 1
    task_track_started: bool = True
    task_time_limit: int = 3600
    task_soft_time_limit: int = 3300


class Settings(BaseSettings):
    """Root settings container."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    regime: RegimeSettings = Field(default_factory=RegimeSettings)
    train: TrainSettings = Field(default_factory=TrainSettings)
    predict: PredictSettings = Field(default_factory=PredictSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    webhook_url: str = Field(default="", alias="WEBHOOK_URL")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the singleton Settings instance.

    Returns:
        Frozen Settings instance loaded from environment.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
