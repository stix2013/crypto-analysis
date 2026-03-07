"""Tests for strategy parameter optimization."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from crypto_analysis.utils.optimization import ParameterOptimizer


class MockStrategy:
    """Mock strategy for testing optimization."""

    def __init__(self, symbols, params):
        self.symbols = symbols
        self.params = params

    def generate_signal(self, data):
        return None


def mock_strategy_factory(symbols: list[str], params: dict[str, Any]) -> MockStrategy:
    """Factory for creating mock strategies."""
    return MockStrategy(symbols, params)


class TestParameterOptimizer:
    """Tests for ParameterOptimizer class."""

    @pytest.fixture
    def mock_data(self):
        """Create mock market data."""
        df = pd.DataFrame(
            {
                "close": np.random.randn(200) + 100,
                "volume": np.random.randn(200) + 1000,
            },
            index=pd.date_range("2023-01-01", periods=200),
        )
        return {"BTC": df}

    def test_initialization(self, mock_data):
        """Test optimizer initialization."""
        optimizer = ParameterOptimizer(
            strategy_factory=mock_strategy_factory,
            data=mock_data,
            symbols=["BTC"],
            initial_equity=5000.0,
        )

        assert optimizer.strategy_factory == mock_strategy_factory
        assert optimizer.symbols == ["BTC"]
        assert optimizer.initial_equity == 5000.0
        assert optimizer.results == []

    def test_grid_search(self, mock_data, monkeypatch):
        """Test grid search parameter sweep."""
        optimizer = ParameterOptimizer(
            strategy_factory=mock_strategy_factory, data=mock_data, symbols=["BTC"]
        )

        param_grid = {"p1": [1, 2], "p2": ["a", "b"]}

        # Mock Backtester.run to return predictable metrics
        from crypto_analysis.signals.backtest import Backtester

        def mock_run(self, start_idx=100):
            return {
                "total_return": self.strategy.params["p1"] * 0.1,
                "sharpe_ratio": self.strategy.params["p1"] * 0.5,
                "max_drawdown": -0.1,
                "num_trades": 5,
                "final_equity": 10000 * (1 + self.strategy.params["p1"] * 0.1),
            }

        monkeypatch.setattr(Backtester, "run", mock_run)

        results_df = optimizer.grid_search(param_grid)

        # 2 * 2 = 4 combinations
        assert len(results_df) == 4
        assert "p1" in results_df.columns
        assert "p2" in results_df.columns
        assert "sharpe_ratio" in results_df.columns

        # Sorted by sharpe_ratio (descending)
        # p1=2 (sharpe=1.0) should be before p1=1 (sharpe=0.5)
        assert results_df.iloc[0]["p1"] == 2
        assert results_df.iloc[-1]["p1"] == 1

    def test_grid_search_exception_handling(self, mock_data, monkeypatch):
        """Test grid search with individual run errors."""
        optimizer = ParameterOptimizer(
            strategy_factory=mock_strategy_factory, data=mock_data, symbols=["BTC"]
        )

        param_grid = {"p1": [1, 2]}

        from crypto_analysis.signals.backtest import Backtester

        def mock_run_fail(self, start_idx=100):
            if self.strategy.params["p1"] == 2:
                raise ValueError("Test error")
            return {"sharpe_ratio": 1.0}

        monkeypatch.setattr(Backtester, "run", mock_run_fail)

        results_df = optimizer.grid_search(param_grid)

        # Only 1 result should be stored (the non-failing one)
        assert len(results_df) == 1
        assert results_df.iloc[0]["p1"] == 1
