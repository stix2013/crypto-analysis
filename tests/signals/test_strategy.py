"""Tests for strategy module - PortfolioManager, Order, Position."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from crypto_analysis.signals.strategy import (
    PortfolioManager,
    Position,
    Order,
    Side,
    OrderType,
    MLStrategy,
    Strategy,
)
from crypto_analysis.signals.base import SignalType


class TestPortfolioManager:
    """Test PortfolioManager class."""

    @pytest.fixture
    def portfolio(self):
        """Create a portfolio manager instance."""
        return PortfolioManager(
            initial_equity=100000.0,
            slippage_pct=0.001,
            commission_rate=0.001,
        )

    @pytest.fixture
    def data_handler(self):
        """Create a mock data handler."""
        handler = MagicMock()
        handler.get_current_price.return_value = 50000.0
        return handler

    def test_portfolio_initialization(self, portfolio):
        """Test portfolio initializes with correct values."""
        assert portfolio.cash == 100000.0
        assert portfolio.initial_equity == 100000.0
        assert portfolio.realized_pnl == 0.0
        assert portfolio.positions == {}

    def test_get_total_equity(self, portfolio, data_handler):
        """Test equity calculation includes cash and position value."""
        portfolio.cash = 80000.0
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )
        data_handler.get_current_price.return_value = 55000.0

        equity = portfolio.get_total_equity(data_handler)
        assert equity == 80000.0 + 55000.0  # cash + position value

    def test_execute_order_new_position(self, portfolio, data_handler):
        """Test executing order creates new position."""
        order = Order(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=1.0,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-01"),
        )

        portfolio.execute_order(order, data_handler)

        assert "BTCUSDT" in portfolio.positions
        pos = portfolio.positions["BTCUSDT"]
        assert pos.size == 1.0
        assert pos.entry_price == 50000.0 * 1.001  # slippage applied

    def test_execute_order_insufficient_funds(self, portfolio, data_handler):
        """Test executing buy order with insufficient cash warns."""
        portfolio.cash = 100.0  # Very low cash
        order = Order(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=10.0,  # Large order that exceeds cash
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-01"),
        )

        with pytest.warns(UserWarning, match="Insufficient funds"):
            portfolio.execute_order(order, data_handler)

    def test_execute_order_increase_long_position(self, portfolio, data_handler):
        """Test adding to existing long position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )
        portfolio.cash = 100000.0

        order = Order(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=0.5,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-02"),
        )

        portfolio.execute_order(order, data_handler)

        pos = portfolio.positions["BTCUSDT"]
        assert pos.size == 1.5
        # VWAP calculation: (1*50000 + 0.5*50000) / 1.5 = 50000

    def test_execute_order_decrease_long_position(self, portfolio, data_handler):
        """Test reducing existing long position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )
        portfolio.cash = 100000.0

        order = Order(
            symbol="BTCUSDT",
            side=Side.SELL,
            size=0.3,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-02"),
        )

        portfolio.execute_order(order, data_handler)

        pos = portfolio.positions["BTCUSDT"]
        assert pos.size == 0.7

    def test_execute_order_close_position_completely(self, portfolio, data_handler):
        """Test closing position completely removes it."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )
        portfolio.cash = 100000.0

        order = Order(
            symbol="BTCUSDT",
            side=Side.SELL,
            size=1.0,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-02"),
        )

        portfolio.execute_order(order, data_handler)

        assert "BTCUSDT" not in portfolio.positions

    def test_execute_order_position_flip_long_to_short(self, portfolio, data_handler):
        """Test flipping from long to short position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )
        portfolio.cash = 100000.0

        order = Order(
            symbol="BTCUSDT",
            side=Side.SELL,
            size=1.5,  # Sell more than we have = flip to short
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-02"),
        )

        portfolio.execute_order(order, data_handler)

        pos = portfolio.positions["BTCUSDT"]
        assert pos.size == -0.5
        assert pos.entry_price == 50000.0 * 0.999  # slippage for sell

    def test_execute_order_short_position(self, portfolio, data_handler):
        """Test executing sell order creates short position."""
        order = Order(
            symbol="BTCUSDT",
            side=Side.SELL,
            size=1.0,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-01"),
        )

        portfolio.execute_order(order, data_handler)

        pos = portfolio.positions["BTCUSDT"]
        assert pos.size == -1.0

    def test_execute_order_cover_short(self, portfolio, data_handler):
        """Test covering a short position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=-1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )
        portfolio.cash = 100000.0

        order = Order(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=0.5,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-02"),
        )

        portfolio.execute_order(order, data_handler)

        pos = portfolio.positions["BTCUSDT"]
        assert pos.size == -0.5

    def test_execute_order_with_stop_loss(self, portfolio, data_handler):
        """Test executing order with stop loss sets position SL."""
        order = Order(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=1.0,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-01"),
            stop_loss=45000.0,
        )

        portfolio.execute_order(order, data_handler)

        pos = portfolio.positions["BTCUSDT"]
        assert pos.stop_loss == 45000.0

    def test_execute_order_with_take_profit(self, portfolio, data_handler):
        """Test executing order with take profit sets position TP."""
        order = Order(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=1.0,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2023-01-01"),
            take_profit=55000.0,
        )

        portfolio.execute_order(order, data_handler)

        pos = portfolio.positions["BTCUSDT"]
        assert pos.take_profit == 55000.0

    def test_get_position_existing(self, portfolio):
        """Test getting existing position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )

        pos = portfolio.get_position("BTCUSDT")
        assert pos is not None
        assert pos.size == 1.0

    def test_get_position_nonexisting(self, portfolio):
        """Test getting non-existent position returns None."""
        pos = portfolio.get_position("ETHUSDT")
        assert pos is None

    def test_check_risk_triggers_stop_loss_long(self, portfolio, data_handler):
        """Test stop loss triggers for long position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
            stop_loss=48000.0,
        )
        data_handler.get_current_price.return_value = 47000.0  # Below SL

        portfolio.check_risk_triggers(data_handler)

        assert "BTCUSDT" not in portfolio.positions

    def test_check_risk_triggers_stop_loss_short(self, portfolio, data_handler):
        """Test stop loss triggers for short position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=-1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
            stop_loss=52000.0,
        )
        data_handler.get_current_price.return_value = 53000.0  # Above SL

        portfolio.check_risk_triggers(data_handler)

        assert "BTCUSDT" not in portfolio.positions

    def test_check_risk_triggers_take_profit_long(self, portfolio, data_handler):
        """Test take profit triggers for long position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
            take_profit=55000.0,
        )
        data_handler.get_current_price.return_value = 56000.0  # Above TP

        portfolio.check_risk_triggers(data_handler)

        assert "BTCUSDT" not in portfolio.positions

    def test_check_risk_triggers_take_profit_short(self, portfolio, data_handler):
        """Test take profit triggers for short position."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=-1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
            take_profit=45000.0,
        )
        data_handler.get_current_price.return_value = 44000.0  # Below TP

        portfolio.check_risk_triggers(data_handler)

        assert "BTCUSDT" not in portfolio.positions

    def test_check_risk_triggers_no_trigger(self, portfolio, data_handler):
        """Test no trigger when price is between SL and TP."""
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
            stop_loss=45000.0,
            take_profit=55000.0,
        )
        data_handler.get_current_price.return_value = 52000.0

        portfolio.check_risk_triggers(data_handler)

        assert "BTCUSDT" in portfolio.positions


class TestOrder:
    """Test Order dataclass."""

    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=1.0,
            order_type=OrderType.LIMIT,
            timestamp=pd.Timestamp("2023-01-01"),
            price=50000.0,
            stop_loss=45000.0,
            take_profit=60000.0,
        )

        assert order.symbol == "BTCUSDT"
        assert order.side == Side.BUY
        assert order.size == 1.0
        assert order.order_type == OrderType.LIMIT
        assert order.price == 50000.0
        assert order.stop_loss == 45000.0
        assert order.take_profit == 60000.0


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            symbol="BTCUSDT",
            size=1.5,
            entry_price=50000.0,
            entry_time=pd.Timestamp("2023-01-01"),
            stop_loss=45000.0,
            take_profit=55000.0,
        )

        assert pos.symbol == "BTCUSDT"
        assert pos.size == 1.5
        assert pos.entry_price == 50000.0


class TestMLStrategy:
    """Test MLStrategy class."""

    @pytest.fixture
    def mock_aggregator(self):
        """Create a mock signal aggregator."""
        agg = MagicMock()
        gen = MagicMock()
        gen.lookback_period = 50
        agg.generators = [gen]
        return agg

    @pytest.fixture
    def strategy(self, mock_aggregator):
        """Create MLStrategy instance."""
        return MLStrategy(
            symbols=["BTCUSDT"],
            aggregator=mock_aggregator,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            kelly_fraction=0.5,
        )

    @pytest.fixture
    def data_handler_with_data(self, sample_ohlcv_data):
        """Create mock data handler with data."""
        handler = MagicMock()
        handler.get_data.return_value = sample_ohlcv_data
        handler.get_current_price.return_value = 30000.0
        return handler

    def test_strategy_initialization(self, strategy, mock_aggregator):
        """Test strategy initializes correctly."""
        assert strategy.symbols == ["BTCUSDT"]
        assert strategy.aggregator is mock_aggregator
        assert strategy.stop_loss_pct == 0.02
        assert strategy.take_profit_pct == 0.04
        assert strategy.kelly_fraction == 0.5

    def test_generate_signals_insufficient_data(
        self, strategy, mock_aggregator, sample_ohlcv_data
    ):
        """Test strategy handles insufficient data gracefully."""
        handler = MagicMock()
        handler.get_data.return_value = sample_ohlcv_data.iloc[
            :10
        ]  # Less than lookback
        portfolio = MagicMock()

        orders = strategy.generate_signals(handler, portfolio)

        assert orders == []

    def test_generate_signals_with_signal(
        self, strategy, data_handler_with_data, mock_aggregator, sample_ohlcv_data
    ):
        """Test strategy generates orders from signals."""
        portfolio = MagicMock()
        portfolio.get_position.return_value = None
        portfolio.get_total_equity.return_value = 100000.0

        mock_signal = MagicMock()
        mock_signal.symbol = "BTCUSDT"
        mock_signal.signal_type = SignalType.ENTRY_LONG
        mock_signal.confidence = 0.8
        mock_signal.timestamp = sample_ohlcv_data.index[-1]

        mock_aggregator.aggregate.return_value = mock_signal

        orders = strategy.generate_signals(data_handler_with_data, portfolio)

        assert len(orders) > 0
        assert orders[0].symbol == "BTCUSDT"

    def test_generate_signals_with_position(
        self, strategy, data_handler_with_data, mock_aggregator, sample_ohlcv_data
    ):
        """Test strategy respects existing positions."""
        portfolio = MagicMock()
        existing_pos = Position(
            symbol="BTCUSDT",
            size=0.5,
            entry_price=30000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )
        portfolio.get_position.return_value = existing_pos
        portfolio.get_total_equity.return_value = 100000.0

        mock_signal = MagicMock()
        mock_signal.symbol = "BTCUSDT"
        mock_signal.signal_type = SignalType.EXIT_LONG
        mock_signal.confidence = 0.8
        mock_signal.timestamp = sample_ohlcv_data.index[-1]

        mock_aggregator.aggregate.return_value = mock_signal

        orders = strategy.generate_signals(data_handler_with_data, portfolio)

        assert len(orders) > 0

    def test_generate_signals_generator_error(
        self, strategy, data_handler_with_data, mock_aggregator, sample_ohlcv_data
    ):
        """Test strategy handles generator errors gracefully."""
        portfolio = MagicMock()
        portfolio.get_position.return_value = None
        portfolio.get_total_equity.return_value = 100000.0

        def generator_error(*args):
            raise ValueError("Generator failed")

        mock_aggregator.generators[0].generate = generator_error

        with pytest.warns(UserWarning, match="Generator .* failed"):
            orders = strategy.generate_signals(data_handler_with_data, portfolio)

        assert orders == []

    def test_signal_to_orders_entry_long(self, strategy, sample_ohlcv_data):
        """Test converting ENTRY_LONG signal to orders."""
        portfolio = MagicMock()
        portfolio.get_total_equity.return_value = 100000.0
        portfolio.get_position.return_value = None

        data_handler = MagicMock()
        data_handler.get_current_price.return_value = 30000.0

        signal = MagicMock()
        signal.symbol = "BTCUSDT"
        signal.signal_type = SignalType.ENTRY_LONG
        signal.confidence = 0.8
        signal.timestamp = sample_ohlcv_data.index[-1]

        orders = strategy._signal_to_orders(signal, portfolio, data_handler)

        assert len(orders) >= 1
        assert orders[0].side == Side.BUY

    def test_signal_to_orders_exit_long(self, strategy, sample_ohlcv_data):
        """Test converting EXIT_LONG signal to orders."""
        portfolio = MagicMock()
        existing_pos = Position(
            symbol="BTCUSDT",
            size=1.0,
            entry_price=30000.0,
            entry_time=pd.Timestamp("2023-01-01"),
        )
        portfolio.get_position.return_value = existing_pos

        data_handler = MagicMock()
        data_handler.get_current_price.return_value = 30000.0

        signal = MagicMock()
        signal.symbol = "BTCUSDT"
        signal.signal_type = SignalType.EXIT_LONG
        signal.confidence = 0.8
        signal.timestamp = sample_ohlcv_data.index[-1]

        orders = strategy._signal_to_orders(signal, portfolio, data_handler)

        assert len(orders) >= 1
        assert orders[0].side == Side.SELL

    def test_signal_to_orders_entry_short_no_position(
        self, strategy, sample_ohlcv_data
    ):
        """Test converting ENTRY_SHORT signal to orders."""
        portfolio = MagicMock()
        portfolio.get_position.return_value = None
        portfolio.get_total_equity.return_value = 100000.0

        data_handler = MagicMock()
        data_handler.get_current_price.return_value = 30000.0

        signal = MagicMock()
        signal.symbol = "BTCUSDT"
        signal.signal_type = SignalType.ENTRY_SHORT
        signal.confidence = 0.8
        signal.timestamp = sample_ohlcv_data.index[-1]

        orders = strategy._signal_to_orders(signal, portfolio, data_handler)

        assert len(orders) >= 1
        assert orders[0].side == Side.SELL

    def test_kelly_sizing(self, strategy):
        """Test Kelly criterion sizing calculation."""
        signal = MagicMock()
        signal.confidence = 0.8
        signal.signal_type = SignalType.ENTRY_LONG

        kelly = strategy._kelly_sizing(signal)

        assert kelly > 0
        assert kelly <= 1.0

    def test_volatility_adjustment(self, strategy):
        """Test volatility adjustment calculation."""
        data_handler = MagicMock()

        vol_adj = strategy._volatility_adjustment("BTCUSDT", data_handler)

        assert vol_adj > 0
