"""Strategy and backtesting integration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from crypto_analysis.signals.aggregator import SignalAggregator
from crypto_analysis.signals.base import Signal, SignalType


class Side(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    """Trading order.

    Attributes:
        symbol: Trading pair symbol
        side: Buy or sell
        size: Order size
        order_type: Market, limit, or stop
        timestamp: Order timestamp
        price: Limit/stop price (optional)
        metadata: Additional order info

    """

    symbol: str
    side: Side
    size: float
    order_type: OrderType
    timestamp: pd.Timestamp
    price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Trading position.

    Attributes:
        symbol: Trading pair symbol
        size: Position size (positive=long, negative=short)
        entry_price: Average entry price
        entry_time: Position entry timestamp
        metadata: Additional position info

    """

    symbol: str
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)


class DataHandler:
    """Abstract data handler for backtesting.

    Handles market data access for strategies.

    """

    def __init__(self) -> None:
        """Initialize data handler."""
        self._data: dict[str, pd.DataFrame] = {}

    def load_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Load OHLCV data for a symbol.

        Args:
            symbol: Trading pair symbol
            data: OHLCV DataFrame

        """
        self._data[symbol] = data.copy()

    def get_data(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """Get historical data for a symbol.

        Args:
            symbol: Trading pair symbol
            lookback: Number of bars to retrieve

        Returns:
            OHLCV DataFrame

        Raises:
            KeyError: If symbol not loaded

        """
        if symbol not in self._data:
            raise KeyError(f"No data loaded for symbol: {symbol}")

        return self._data[symbol].tail(lookback)

    def get_current_price(self, symbol: str) -> float:
        """Get current market price.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price

        """
        data = self.get_data(symbol, lookback=1)
        return float(data["close"].iloc[-1])


class PortfolioManager:
    """Portfolio manager for backtesting.

    Tracks positions, equity, and P&L.

    """

    def __init__(self, initial_equity: float = 10000.0) -> None:
        """Initialize portfolio manager.

        Args:
            initial_equity: Starting equity

        """
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.positions: dict[str, Position] = {}
        self.cash = initial_equity
        self.orders: list[Order] = []

    def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position or None if not held

        """
        return self.positions.get(symbol)

    def get_total_equity(self, data_handler: DataHandler) -> float:
        """Calculate total portfolio equity.

        Args:
            data_handler: Data handler for prices

        Returns:
            Total equity value

        """
        position_value = 0.0
        for symbol, position in self.positions.items():
            current_price = data_handler.get_current_price(symbol)
            position_value += position.size * current_price

        return self.cash + position_value

    def execute_order(self, order: Order, data_handler: DataHandler) -> None:
        """Execute an order.

        Args:
            order: Order to execute
            data_handler: Data handler for prices

        """
        price = data_handler.get_current_price(order.symbol)
        cost = order.size * price

        if order.side == Side.BUY:
            self.cash -= cost
        else:
            self.cash += cost

        self.orders.append(order)

        # Update position
        current_position = self.positions.get(order.symbol)
        if current_position:
            # Calculate new average entry price
            total_size = current_position.size + (
                order.size if order.side == Side.BUY else -order.size
            )
            if total_size == 0:
                del self.positions[order.symbol]
            else:
                current_position.size = total_size
                current_position.entry_price = price
        else:
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                size=order.size if order.side == Side.BUY else -order.size,
                entry_price=price,
                entry_time=order.timestamp,
            )


class Strategy:
    """Base strategy class.

    All strategies must inherit from this class.

    """

    def __init__(self, symbols: list[str]) -> None:
        """Initialize strategy.

        Args:
            symbols: List of symbols to trade

        """
        self.symbols = symbols

    def generate_signals(
        self,
        data_handler: DataHandler,
        portfolio: PortfolioManager,
    ) -> list[Order]:
        """Generate trading orders.

        Args:
            data_handler: Data handler for market data
            portfolio: Portfolio manager

        Returns:
            List of orders to execute

        """
        return []


class MLStrategy(Strategy):
    """Strategy that uses multiple signal generators with aggregation.

    Combines ML, technical, and statistical signal generators
    to produce trading decisions.

    Attributes:
        aggregator: SignalAggregator for combining signals
        symbols: List of symbols to trade

    """

    def __init__(self, symbols: list[str], aggregator: SignalAggregator) -> None:
        """Initialize ML strategy.

        Args:
            symbols: List of symbols to trade
            aggregator: SignalAggregator instance

        """
        super().__init__(symbols)
        self.aggregator = aggregator

    def generate_signals(
        self,
        data_handler: DataHandler,
        portfolio: PortfolioManager,
    ) -> list[Order]:
        """Generate trading orders from aggregated signals.

        Args:
            data_handler: Data handler for market data
            portfolio: Portfolio manager

        Returns:
            List of orders to execute

        """
        import warnings

        orders = []

        for symbol in self.symbols:
            try:
                # Get enough data for all generators
                max_lookback = max(g.lookback_period for g in self.aggregator.generators)
                data = data_handler.get_data(symbol, lookback=max_lookback + 50)

                if len(data) < max_lookback:
                    continue

                # Get current position
                position = portfolio.get_position(symbol)
                current_size = position.size if position else 0

                # Generate signals from all generators
                all_signals = []
                for generator in self.aggregator.generators:
                    try:
                        sigs = generator.generate(data, current_size)
                        all_signals.extend(sigs)
                    except Exception as e:
                        warnings.warn(f"Generator {generator.name} failed: {e}")

                # Aggregate
                final_signal = self.aggregator.aggregate(all_signals, current_size)

                if final_signal:
                    orders.extend(self._signal_to_orders(final_signal, portfolio, data_handler))

            except Exception as e:
                warnings.warn(f"Strategy error for {symbol}: {e}")
                continue

        return orders

    def _signal_to_orders(
        self,
        signal: Signal,
        portfolio: PortfolioManager,
        data_handler: DataHandler,
    ) -> list[Order]:
        """Convert signal to order(s).

        Args:
            signal: Signal to convert
            portfolio: Portfolio manager
            data_handler: Data handler

        Returns:
            List of orders

        """
        orders = []

        current_price = data_handler.get_current_price(signal.symbol)
        position = portfolio.get_position(signal.symbol)
        current_size = position.size if position else 0

        # Determine order parameters based on signal
        if signal.signal_type == SignalType.ENTRY_LONG:
            if current_size < 0:
                # Close short first
                orders.append(
                    Order(
                        symbol=signal.symbol,
                        side=Side.BUY,
                        size=abs(current_size),
                        order_type=OrderType.MARKET,
                        timestamp=signal.timestamp,
                    )
                )

            # Calculate position size based on confidence and Kelly criterion
            equity = portfolio.get_total_equity(data_handler)
            kelly_fraction = self._kelly_sizing(signal)
            base_size = (equity * 0.1 * kelly_fraction) / current_price

            orders.append(
                Order(
                    symbol=signal.symbol,
                    side=Side.BUY,
                    size=base_size,
                    order_type=OrderType.MARKET,
                    timestamp=signal.timestamp,
                )
            )

        elif signal.signal_type == SignalType.ENTRY_SHORT:
            if current_size > 0:
                orders.append(
                    Order(
                        symbol=signal.symbol,
                        side=Side.SELL,
                        size=current_size,
                        order_type=OrderType.MARKET,
                        timestamp=signal.timestamp,
                    )
                )

            equity = portfolio.get_total_equity(data_handler)
            kelly_fraction = self._kelly_sizing(signal)
            base_size = (equity * 0.1 * kelly_fraction) / current_price

            orders.append(
                Order(
                    symbol=signal.symbol,
                    side=Side.SELL,
                    size=base_size,
                    order_type=OrderType.MARKET,
                    timestamp=signal.timestamp,
                )
            )

        elif signal.signal_type in [
            SignalType.EXIT_LONG,
            SignalType.EXIT_SHORT,
            SignalType.RISK_OFF,
        ]:
            if current_size != 0:
                orders.append(
                    Order(
                        symbol=signal.symbol,
                        side=Side.SELL if current_size > 0 else Side.BUY,
                        size=abs(current_size),
                        order_type=OrderType.MARKET,
                        timestamp=signal.timestamp,
                    )
                )

        return orders

    def _kelly_sizing(self, signal: Signal) -> float:
        """Calculate Kelly Criterion fraction based on confidence.

        Args:
            signal: Signal with confidence score

        Returns:
            Position size fraction (0 to 1)

        """
        # Simplified Kelly: f = (p*b - q) / b
        # where p = win probability (confidence), b = win/loss ratio
        p = signal.confidence
        b = 1.5  # Assumed profit/loss ratio
        q = 1 - p

        kelly = (p * b - q) / b if b != 0 else 0

        # Use half-Kelly for safety
        return max(0, min(kelly / 2, 1.0))
