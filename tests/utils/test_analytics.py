"""Tests for performance analytics."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from crypto_analysis.utils.analytics import PerformanceAnalyzer


@dataclass
class MockOrder:
    symbol: str
    metadata: dict[str, Any]


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer class."""

    def test_calculate_metrics_empty(self):
        """Test with empty data."""
        equity = pd.Series([], dtype=float)
        metrics = PerformanceAnalyzer.calculate_metrics(equity, [])
        assert metrics == {}

    def test_calculate_metrics_basic(self):
        """Test basic metric calculations."""
        # 10 days of data, 10% total return
        equity = pd.Series(
            [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 108.0, 110.0],
            index=pd.date_range("2023-01-01", periods=10),
        )

        # Mock 2 trades: one win (+10), one loss (-5)
        orders = [
            MockOrder(symbol="BTC", metadata={"pnl": 10.0}),
            MockOrder(symbol="BTC", metadata={"pnl": -5.0}),
            MockOrder(symbol="ETH", metadata={}),  # No PnL
        ]

        metrics = PerformanceAnalyzer.calculate_metrics(
            equity,
            orders,
            periods_per_year=252,  # Daily
        )

        assert metrics["total_return"] == pytest.approx(0.1)
        assert metrics["num_trades"] == 2
        assert metrics["win_rate"] == 0.5
        assert metrics["profit_factor"] == pytest.approx(2.0)
        assert metrics["avg_pnl"] == 2.5
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert metrics["num_symbols"] == 2

    def test_calculate_trade_metrics_empty(self):
        """Test trade metrics with no PnL orders."""
        orders = [MockOrder(symbol="BTC", metadata={})]
        metrics = PerformanceAnalyzer._calculate_trade_metrics(orders)

        assert metrics["num_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0

    def test_annual_return_calculation(self):
        """Test annualized return for different periods."""
        # 2 years of data, 21% total return (approx 10% CAGR)
        equity = pd.Series(
            [100.0] * 504 + [121.0], index=pd.date_range("2021-01-01", periods=505)
        )

        metrics = PerformanceAnalyzer.calculate_metrics(
            equity, [], periods_per_year=252
        )

        # (1 + 0.21)^(1/2) - 1 = 1.1 - 1 = 0.1
        assert metrics["annual_return"] == pytest.approx(0.1, rel=1e-2)

    def test_plot_equity_curve_no_matplotlib(self, monkeypatch):
        """Test plot handling when matplotlib is missing."""
        import sys

        # Simulate missing matplotlib
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

        equity_df = pd.DataFrame({"equity": [100, 110]}, index=[0, 1])
        # This shouldn't crash
        fig = PerformanceAnalyzer.plot_equity_curve(equity_df)
        assert fig is None

    def test_plot_equity_curve_with_matplotlib(self):
        """Test plot equity curve with matplotlib."""
        pytest.importorskip("matplotlib")

        equity_df = pd.DataFrame(
            {"equity": [100, 110, 105, 120]},
            index=pd.date_range("2023-01-01", periods=4),
        )

        fig = PerformanceAnalyzer.plot_equity_curve(equity_df)
        assert fig is not None

    def test_plot_equity_curve_with_title(self):
        """Test plot equity curve with custom title."""
        pytest.importorskip("matplotlib")

        equity_df = pd.DataFrame(
            {"equity": [100, 110, 105, 120]},
            index=pd.date_range("2023-01-01", periods=4),
        )

        fig = PerformanceAnalyzer.plot_equity_curve(equity_df, title="Custom Title")
        assert fig is not None
