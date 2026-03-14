# Refactor Plan: Centralize Settings from .env

## Design Decisions (Confirmed)
- **Frozen dataclasses**: Settings are immutable after creation
- **pydantic-settings**: For validation and type coercion (add to dependencies)
- **Strict singleton**: Tests must mock `get_settings()` or use `Settings.from_env()` directly

## Overview
Consolidate all environment variable access into a single `Settings` class. Remove all `os.getenv()` calls from source code except the centralized settings module. This improves testability, single source of truth, and explicit dependency injection.

## Files Affected

### Source Files with `os.getenv()`:
1. **`src/crypto_analysis/data/binance.py`** (lines 28-30)
   - `BINANCE_API_KEY`, `BINANCE_SECRET_KEY` in `BinanceConfig.__post_init__`

2. **`src/crypto_analysis/online/generator.py`** (lines 510-517)
   - `REGIME_THRESHOLD_BASE`, `REGIME_ADJUST_TRENDING_UP`, etc. in `_get_regime_threshold()`

### Scripts using `dotenv.load_dotenv()`:
3. **`scripts/train_online.py`** (lines 20-22, 32-62)
   - Uses `os.environ.get()` for defaults like `TRAIN_SYMBOL`, `TRAIN_INTERVAL`, etc.

4. **`scripts/predict.py`** (lines 14-16)
   - Uses `dotenv.load_dotenv()` for env loading

5. **`worker/tasks.py`** (lines 7, 18)
   - Uses `dotenv.load_dotenv()` and needs env vars for training/prediction

---

## Implementation Steps

### Step 1: Create Settings Class
**File**: `src/crypto_analysis/settings.py`

Create a centralized settings class using `pydantic-settings` with frozen dataclasses:

```python
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

    threshold_base: float = Field(default=0.10, alias="THRESHOLD_BASE")
    adjust_trending_up: float = Field(default=0.05, alias="ADJUST_TRENDING_UP")
    adjust_trending_down: float = Field(default=0.05, alias="ADJUST_TRENDING_DOWN")
    adjust_ranging: float = Field(default=0.20, alias="ADJUST_RANGING")
    adjust_volatile: float = Field(default=0.15, alias="ADJUST_VOLATILE")
    adjust_crash: float = Field(default=0.30, alias="ADJUST_CRASH")


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

    broker_url: str = Field(default="redis://localhost:6379/0", alias="BROKER_URL")
    result_backend: str = Field(default="redis://localhost:6379/0", alias="RESULT_BACKEND")
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    worker_concurrency: int = Field(default=1, alias="WORKER_CONCURRENCY")
    task_track_started: bool = Field(default=True, alias="TASK_TRACK_STARTED")
    task_time_limit: int = Field(default=3600, alias="TASK_TIME_LIMIT")
    task_soft_time_limit: int = Field(default=3300, alias="TASK_SOFT_TIME_LIMIT")


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


# Singleton instance
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
```

**Note**: pydantic-settings automatically:
- Loads from `.env` file
- Performs type coercion (e.g., `"5000"` → `5000`)
- Validates types at runtime

---

### Step 2: Update `src/crypto_analysis/data/binance.py`

**Changes**:
1. Remove `import os` and `import dotenv` and `dotenv.load_dotenv()`
2. Import `get_settings` and use it in `BinanceConfig`

```python
from crypto_analysis.settings import get_settings

@dataclass
class BinanceConfig:
    """Configuration for Binance API connection."""

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://fapi.binance.com"
    testnet: bool = False

    def __post_init__(self) -> None:
        settings = get_settings()
        if not self.api_key:
            object.__setattr__(self, "api_key", settings.binance.api_key)
        if not self.secret_key:
            object.__setattr__(self, "secret_key", settings.binance.secret_key)
```

**Note**: Uses `object.__setattr__` because nested settings are frozen.

---

### Step 3: Update `src/crypto_analysis/online/generator.py`

**Changes**:
1. Remove `import os` (line 4)
2. Import `get_settings` and use it instead of `os.getenv()`

```python
from crypto_analysis.settings import get_settings

def _get_regime_threshold(self, regime: MarketRegime) -> float:
    """Adjust signal threshold based on market regime.

    Args:
        regime: Current market regime

    Returns:
        Adjusted threshold value
    """
    settings = get_settings()
    base_threshold = settings.regime.threshold_base

    adjustments = {
        "trending_up": settings.regime.adjust_trending_up,
        "trending_down": settings.regime.adjust_trending_down,
        "ranging": settings.regime.adjust_ranging,
        "volatile": settings.regime.adjust_volatile,
        "crash": settings.regime.adjust_crash,
    }

    return base_threshold + adjustments.get(regime.name, 0.1)
```

---

### Step 4: Update `scripts/train_online.py`

**Changes**:
1. Remove `load_dotenv()` call and `import os`
2. Import `Settings` and use for defaults

```python
from crypto_analysis.settings import get_settings

def main() -> None:
    settings = get_settings()

    parser.add_argument(
        "symbol",
        nargs="?",
        default=settings.train.symbol,
        help="Trading pair symbol",
    )
    parser.add_argument(
        "--interval",
        default=settings.train.interval,
        help="Kline interval",
    )
    # ... etc
```

---

### Step 5: Update `scripts/predict.py`

**Changes**:
1. Remove `load_dotenv()` call
2. Settings module handles `.env` loading

---

### Step 6: Update `worker/tasks.py`

**Changes**:
1. Remove `from dotenv import load_dotenv` and `load_dotenv()`
2. Settings module handles env loading

---

### Step 7: Create `tests/test_settings.py`

New test file for the Settings module:

```python
"""Tests for centralized settings module."""
import pytest
from crypto_analysis.settings import Settings, BinanceSettings, RegimeSettings, get_settings


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
        settings = BinanceSettings(api_key="key")
        with pytest.raises(Exception):  # ValidationError from pydantic
            settings.api_key = "new_key"  # type: ignore[misc]


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

    def test_mocking_for_tests(self, monkeypatch):
        # Tests can override env vars before calling get_settings
        # or create Settings instance directly
        custom_settings = Settings(
            binance=BinanceSettings(api_key="custom_key")
        )
        assert custom_settings.binance.api_key == "custom_key"
```

---

### Step 8: Update `tests/data/test_binance.py`

Update tests to use Settings injection:

```python
from crypto_analysis.settings import BinanceSettings, get_settings

class TestBinanceConfig:
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

    def test_env_vars_via_settings(self, monkeypatch):
        """Test loading from environment variables via Settings."""
        monkeypatch.setenv("BINANCE_API_KEY", "env_api_key")
        monkeypatch.setenv("BINANCE_SECRET_KEY", "env_secret_key")
        # BinanceConfig uses get_settings() internally
        config = BinanceConfig()
        assert config.api_key == "env_api_key"
        assert config.secret_key == "env_secret_key"
```

**Note**: The singleton pattern means tests that modify env vars should avoid calling `get_settings()` before setting env vars, or create `BinanceConfig` with explicit values.

---

### Step 9: Export Settings from `__init__.py`

**File**: `src/crypto_analysis/__init__.py`

Add to imports:
```python
from crypto_analysis.settings import Settings, get_settings
```

---

## Testing Checklist

1. Run all tests: `pytest`
2. Run linting: `ruff check src/ && mypy src/`
3. Test training script: `python scripts/train_online.py ETHUSDT --bars 100 --warmup-bars 50`
4. Test worker: `./docker-manage.sh up && pytest tests/`

---

## Dependency Changes

Add `pydantic-settings` to `pyproject.toml`:

```toml
dependencies = [
    ...
    "pydantic-settings>=2.0.0",
    ...
]
```

---

## Notes

- **Immutable**: All settings classes are frozen (cannot be modified after creation)
- **Type Coercion**: pydantic-settings handles string-to-int/float conversion automatically
- **Singleton**: `get_settings()` returns the same instance; tests should create `Settings()` directly or set env vars before first call
- **Backward Compatibility**: All existing `.env` variables work unchanged
