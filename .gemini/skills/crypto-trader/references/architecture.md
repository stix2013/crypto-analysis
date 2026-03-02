# Crypto Analysis Project Architecture

The project is designed for real-time cryptocurrency trading signal generation and backtesting, with a focus on online learning and adaptive models.

## Core Components

### 1. Data Layer (`src/crypto_analysis/data/`)
- `binance.py`: Client for fetching OHLCV data from Binance Futures API. Handles rate limiting and historical data retrieval.

### 2. Signal Generation Layer (`src/crypto_analysis/signals/`)
- `base.py`: Defines `Signal`, `SignalType`, `Indicator`, and `SignalGenerator` base classes.
- `technical.py`: Implements common technical indicators (RSI, MACD, etc.) and technical signal generators.
- `ml_generators.py`: Bridges ML models with the signal generation system.
- `statistical.py`: Signal generation based on statistical methods like z-scores, cointegration, etc.
- `aggregator.py`: Combines multiple signals using different aggregation logic (weighted, majority, etc.).

### 3. Strategy & Execution Layer (`src/crypto_analysis/signals/strategy.py`)
- `Strategy`: Base class for defining trading logic.
- `MLStrategy`: High-level strategy that uses aggregated signals to make decisions.
- `PortfolioManager`: Tracks positions, equity, and executes simulated orders.
- `DataHandler`: Manages market data access for strategies.

### 4. Online Learning Layer (`src/crypto_analysis/online/`)
- `pipeline.py`: Orchestrates data fetching, pre-processing, model training, and signal generation in a continuous loop.
- `generator.py`: Specific signal generators for online models.
- `models/`: Implementations of online models (`OnlineLSTM`, `OnlineRF`, `OnlineNN`).
- `detection/`: Market regime detection and adaptive parameter tuning (`AdaptiveLR`).

## Workflow

1. **Data Ingestion**: `BinanceClient` fetches recent OHLCV data.
2. **Indicator Calculation**: `Indicator` objects calculate features from raw data.
3. **Signal Generation**: `SignalGenerator` objects produce buy/sell signals based on features.
4. **Signal Aggregation**: `SignalAggregator` combines signals from multiple sources.
5. **Strategy Execution**: `MLStrategy` receives aggregated signals and generates `Order` objects.
6. **Portfolio Update**: `PortfolioManager` executes orders and updates positions.
7. **Online Update**: `OnlinePipeline` updates models with the latest data and performance feedback.
