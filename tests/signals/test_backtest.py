"""Tests for Backtesting engine."""

import numpy as np
import pandas as pd
from crypto_analysis.signals.backtest import Backtester
from crypto_analysis.signals.strategy import (
    DataHandler,
    Order,
    OrderType,
    PortfolioManager,
    Side,
    Strategy,
)


class DummyStrategy(Strategy):
    """Simple strategy for testing backtester."""

    def generate_signals(self, data_handler, portfolio):
        symbol = self.symbols[0]
        data = data_handler.get_data(symbol)

        # Simple crossover strategy
        ma5 = data["close"].rolling(5).mean().iloc[-1]
        ma20 = data["close"].rolling(20).mean().iloc[-1]

        pos = portfolio.get_position(symbol)

        if ma5 > ma20 and (not pos or pos.size <= 0):
            return [Order(symbol, Side.BUY, 1.0, OrderType.MARKET, data.index[-1])]
        elif ma5 < ma20 and pos and pos.size > 0:
            return [Order(symbol, Side.SELL, 1.0, OrderType.MARKET, data.index[-1])]
        return []


def create_test_data(n=200):
    """Create trending test data."""
    np.random.seed(42)
    # Start at 100, upward trend
    prices = 100 + np.cumsum(np.random.randn(n) + 0.1)
    df = pd.DataFrame(
        {
            "open": prices - 0.5,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": np.random.uniform(1000, 5000, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="h"),
    )
    return df


class TestBacktester:
    """Tests for Backtester class."""

    def test_backtester_initialization(self):
        data = {"BTC": create_test_data()}
        strategy = DummyStrategy(symbols=["BTC"])
        backtester = Backtester(strategy, data)

        assert backtester.portfolio.initial_equity == 10000.0
        assert backtester.strategy == strategy

    def test_backtester_run(self):
        data = {"BTC": create_test_data(200)}
        strategy = DummyStrategy(symbols=["BTC"])
        backtester = Backtester(strategy, data)

        results = backtester.run(start_idx=50)

        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert len(backtester.equity_history) == 150
        assert results["num_trades"] > 0

    def test_backtester_metrics(self):
        data = {"BTC": create_test_data(100)}
        strategy = DummyStrategy(symbols=["BTC"])
        backtester = Backtester(strategy, data)

        # Manually populate history to test metrics
        for i in range(10):
            backtester.equity_history.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=i),
                    "equity": 10000.0 * (1.01**i),
                    "cash": 10000.0,
                    "positions": {},
                }
            )

        results = backtester._calculate_results()
        assert results["total_return"] > 0
        assert results["sharpe_ratio"] > 0
        assert results["max_drawdown"] == 0.0  # Straight up

    def test_backtester_short_pnl(self):
        """Test that short PnL is calculated correctly in portfolio equity."""
        data_handler = DataHandler()
        portfolio = PortfolioManager(
            initial_equity=10000.0, commission_rate=0, slippage_pct=0
        )

        # 1. Load data where price goes DOWN
        df = pd.DataFrame(
            {"close": [100, 90]}, index=pd.date_range("2023-01-01", periods=2, freq="h")
        )

        data_handler.load_data("BTC", df.iloc[:1])

        # 2. Enter short at 100
        order = Order("BTC", Side.SELL, 10, OrderType.MARKET, df.index[0])
        portfolio.execute_order(order, data_handler)

        # Initial cash was 10000, sold 10 at 100 -> cash = 11000
        assert portfolio.cash == 11000.0
        assert portfolio.positions["BTC"].size == -10

        # 3. Price goes down to 90
        data_handler.load_data("BTC", df)
        equity = portfolio.get_total_equity(data_handler)

        # Margin = 10 * 100 = 1000
        # Unrealized PnL = (100 - 90) * 10 = 100
        # Total equity should be initial (10000) + PnL (100) = 10100
        # My implementation: cash (11000) + position_value
        # position_value = margin (1000) + unrealized_pnl (100) wait...
        # If I sold for 1000, and now it costs 900 to buy back, I made 100.
        # Cash is 11000. Liability is 900. Equity = 11000 - 900 = 10100.

        # Let's check my formula:
        # cash + (-current_price * abs(size)) was my old one.
        # 11000 + (-90 * 10) = 10100.

        # My NEW formula:
        # margin = abs(size) * entry_price = 10 * 100 = 1000
        # unrealized_pnl = (entry_price - current_price) * abs(size) = (100 - 90) * 10 = 100
        # position_value = margin + unrealized_pnl = 1100 ??? No.

        # Wait, if I add 'margin' back to cash?
        # When I SELL, cash increases by 'cost'.
        # If I bought, cash decreases by 'cost'.

        # Let's re-think PortfolioManager.cash
        # For Long: cash decreases by cost. Equity = cash + current_price * size.
        # For Short: cash INCREASES by cost. Equity = cash - current_price * abs(size).

        assert equity == 10100.0

    def test_stop_loss_trigger(self):
        """Test that Stop-Loss correctly triggers an exit."""
        data_handler = DataHandler()
        portfolio = PortfolioManager(
            initial_equity=10000.0, commission_rate=0, slippage_pct=0
        )

        # 1. Price goes DOWN (against long)
        df = pd.DataFrame(
            {
                "close": [100, 97]  # 3% drop
            },
            index=pd.date_range("2023-01-01", periods=2, freq="h"),
        )

        data_handler.load_data("BTC", df.iloc[:1])

        # 2. Enter long with 2% SL
        order = Order(
            "BTC", Side.BUY, 10, OrderType.MARKET, df.index[0], stop_loss=98.0
        )
        portfolio.execute_order(order, data_handler)

        assert len(portfolio.positions) == 1

        # 3. Check triggers with new price (97)
        data_handler.load_data("BTC", df)
        portfolio.check_risk_triggers(data_handler)

        # Should be closed
        assert len(portfolio.positions) == 0
        assert portfolio.orders[-1].metadata["reason"] == "stop_loss"
        assert portfolio.orders[-1].side == Side.SELL

    def test_backtester_process_signal(self):
        """Test the simple process_signal API used by worker tasks."""
        backtester = Backtester(
            initial_capital=10000.0,
            commission=0.0004,
            slippage_pct=0.0,
        )

        timestamp1 = pd.Timestamp("2023-01-01 00:00:00")
        timestamp2 = pd.Timestamp("2023-01-01 01:00:00")
        timestamp3 = pd.Timestamp("2023-01-01 02:00:00")

        # 1. Buy at 100
        backtester.process_signal(timestamp1, "BTCUSDT", "BUY", 100.0)
        assert len(backtester.portfolio.positions) == 1
        assert backtester.portfolio.positions["BTCUSDT"].size == 1.0
        # Cash was 10000. Bought 1.0 at 100.0 with 0.0004 commission.
        # Cost = 100 * 1.0 * (1 + 0.0004) = 100.04
        # New cash = 10000 - 100.04 = 9899.96
        assert abs(backtester.portfolio.cash - 9899.96) < 0.01

        # 2. Exit at 110 (profitable trade)
        backtester.process_signal(timestamp2, "BTCUSDT", "EXIT", 110.0)
        # Revenue = 110 * 1.0 * (1 - 0.0004) = 109.956
        # New cash = 9899.96 + 109.956 = 10009.916
        assert len(backtester.portfolio.positions) == 0
        assert abs(backtester.portfolio.cash - 10009.916) < 0.01

        # 3. Check equity curve and trades
        equity_curve = backtester.get_equity_curve()
        trades = backtester.get_trades()

        assert len(equity_curve) == 2
        assert len(trades) == 2
        assert trades.iloc[0]["side"] == "BUY"
        assert trades.iloc[1]["side"] == "SELL"

    def test_backtester_flip_and_exit(self):
        """Test flipping positions and EXIT signal in process_signal."""
        backtester = Backtester(
            initial_capital=10000.0,
            commission=0,
            slippage_pct=0.0,
        )

        # 1. Buy 1.0 at 100
        backtester.process_signal(pd.Timestamp("2023-01-01 00:00:00"), "BTC", "BUY", 100.0)
        assert backtester.portfolio.positions["BTC"].size == 1.0
        assert backtester.portfolio.cash == 9900.0

        # 2. Sell (flip to SHORT 1.0) at 110
        # To go from LONG 1.0 to SHORT 1.0, we need to SELL 2.0.
        backtester.process_signal(pd.Timestamp("2023-01-01 01:00:00"), "BTC", "SELL", 110.0)
        assert backtester.portfolio.positions["BTC"].size == -1.0
        # Cash was 9900. Sold 2.0 at 110.0. Revenue = 220.0
        # New cash = 9900 + 220 = 10120.0
        assert backtester.portfolio.cash == 10120.0

        # 3. EXIT at 105
        # To go from SHORT 1.0 to 0.0, we need to BUY 1.0.
        backtester.process_signal(pd.Timestamp("2023-01-01 02:00:00"), "BTC", "EXIT", 105.0)
        assert len(backtester.portfolio.positions) == 0
        # Cash was 10120. Bought 1.0 at 105.0. Cost = 105.0
        # New cash = 10120 - 105 = 10015.0
        assert backtester.portfolio.cash == 10015.0
