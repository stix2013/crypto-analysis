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
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        metadata: Additional order info

    """

    symbol: str
    side: Side
    size: float
    order_type: OrderType
    timestamp: pd.Timestamp
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Trading position.

    Attributes:
        symbol: Trading pair symbol
        size: Position size (positive=long, negative=short)
        entry_price: Average entry price
        entry_time: Position entry timestamp
        stop_loss: Active stop loss price
        take_profit: Active take profit price
        metadata: Additional position info

    """

    symbol: str
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float | None = None
    take_profit: float | None = None
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

    def __init__(
        self,
        initial_equity: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        """Initialize portfolio manager.

        Args:
            initial_equity: Starting equity

        """
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.positions: dict[str, Position] = {}
        self.cash = initial_equity
        self.orders: list[Order] = []
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.realized_pnl = 0.0
        self.equity_history: list[dict[str, Any]] = []

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
            if position.size > 0:
                # Long position: cash decreased by cost.
                # value = current_price * size
                position_value += current_price * position.size
            else:
                # Short position: cash increased by cost.
                # liability = current_price * abs(size)
                position_value -= current_price * abs(position.size)

        return self.cash + position_value

    def execute_order(self, order: Order, data_handler: DataHandler) -> None:
        """Execute an order with slippage, commission, VWAP, and margin checks.

        Args:
            order: Order to execute
            data_handler: Data handler for prices

        """
        base_price = data_handler.get_current_price(order.symbol)

        # Apply slippage
        if order.side == Side.BUY:
            price = base_price * (1 + self.slippage_pct)
        else:
            price = base_price * (1 - self.slippage_pct)

        cost = order.size * price
        commission = cost * self.commission_rate

        # Margin Check (Simple)
        if order.side == Side.BUY and (self.cash - cost - commission) < 0:
            import warnings

            warnings.warn(f"Insufficient funds to execute BUY order for {order.symbol}.")
            return

        # Deduct commission
        self.cash -= commission

        if order.side == Side.BUY:
            self.cash -= cost
        else:
            self.cash += cost

        # Update order with execution details
        order.price = price
        order.metadata["commission"] = commission
        order.metadata["slippage_price"] = price
        self.orders.append(order)

        # Update position
        current_position = self.positions.get(order.symbol)

        if current_position:
            is_long = current_position.size > 0
            is_buy = order.side == Side.BUY

            # Update SL/TP if provided in new order
            if order.stop_loss:
                current_position.stop_loss = order.stop_loss
            if order.take_profit:
                current_position.take_profit = order.take_profit

            # Increasing position (Buy when Long, Sell when Short)
            if (is_long and is_buy) or (not is_long and not is_buy):
                new_size = current_position.size + (order.size if is_buy else -order.size)
                # VWAP calculation
                current_value = abs(current_position.size) * current_position.entry_price
                new_value = order.size * price
                new_entry_price = (current_value + new_value) / abs(new_size)

                current_position.size = new_size
                current_position.entry_price = new_entry_price

            # Decreasing position (Sell when Long, Buy when Short)
            else:
                # Calculate realized PnL
                if is_long:  # Selling a long position
                    pnl = (price - current_position.entry_price) * order.size
                else:  # Buying to cover a short position
                    pnl = (current_position.entry_price - price) * order.size

                self.realized_pnl += pnl

                new_size = current_position.size + (order.size if is_buy else -order.size)

                if abs(new_size) < 1e-8:  # Floating point zero
                    del self.positions[order.symbol]
                elif (is_long and new_size < 0) or (not is_long and new_size > 0):
                    # Position flipped
                    del self.positions[order.symbol]
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        size=new_size,
                        entry_price=price,
                        entry_time=order.timestamp,
                    )
                else:
                    # Partial close (entry price doesn't change)
                    current_position.size = new_size
        else:
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                size=order.size if order.side == Side.BUY else -order.size,
                entry_price=price,
                entry_time=order.timestamp,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )

    def check_risk_triggers(self, data_handler: DataHandler) -> None:
        """Check all positions for Stop-Loss or Take-Profit triggers.

        Args:
            data_handler: Data handler for current prices
        """
        symbols_to_close = []

        for symbol, pos in self.positions.items():
            price = data_handler.get_current_price(symbol)

            # Stop Loss
            if pos.stop_loss:
                if (pos.size > 0 and price <= pos.stop_loss) or (
                    pos.size < 0 and price >= pos.stop_loss
                ):
                    symbols_to_close.append((symbol, "stop_loss", price))
                    continue

            # Take Profit
            if pos.take_profit:
                if (pos.size > 0 and price >= pos.take_profit) or (
                    pos.size < 0 and price <= pos.take_profit
                ):
                    symbols_to_close.append((symbol, "take_profit", price))

        for symbol, reason, price in symbols_to_close:
            pos = self.positions[symbol]
            side = Side.SELL if pos.size > 0 else Side.BUY
            order = Order(
                symbol=symbol,
                side=side,
                size=abs(pos.size),
                order_type=OrderType.MARKET,
                timestamp=pd.Timestamp.now(),  # In backtest this will be the current bar time
                metadata={"reason": reason, "trigger_price": price},
            )
            self.execute_order(order, data_handler)


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

        # Max absolute position size limit (e.g. 10% of equity per trade * leverage)
        equity = portfolio.get_total_equity(data_handler)
        kelly_fraction = self._kelly_sizing(signal)

        # Volatility-adjusted sizing: adjust target risk based on recent ATR
        # Base allocation = 20% of equity
        # Adjusted = Base * (Historical Vol / Current Vol)
        vol_adj = self._volatility_adjustment(signal.symbol, data_handler)

        target_size = (
            equity * 0.2 * kelly_fraction * vol_adj
        ) / current_price  # Using 20% max equity allocation per asset

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
                current_size = 0

            # Only add to position if we are below target size
            if current_size < target_size:
                size_to_buy = target_size - current_size
                # Require at least some minimum size to trade
                if size_to_buy > (equity * 0.01 / current_price):
                    orders.append(
                        Order(
                            symbol=signal.symbol,
                            side=Side.BUY,
                            size=size_to_buy,
                            order_type=OrderType.MARKET,
                            timestamp=signal.timestamp,
                            stop_loss=current_price * 0.98,
                            take_profit=current_price * 1.04,
                        )
                    )

        elif signal.signal_type == SignalType.ENTRY_SHORT:
            if current_size > 0:
                # Close long first
                orders.append(
                    Order(
                        symbol=signal.symbol,
                        side=Side.SELL,
                        size=current_size,
                        order_type=OrderType.MARKET,
                        timestamp=signal.timestamp,
                    )
                )
                current_size = 0

            # Target size for short is negative
            if current_size > -target_size:
                size_to_sell = abs(target_size) - abs(current_size)
                if size_to_sell > (equity * 0.01 / current_price):
                    orders.append(
                        Order(
                            symbol=signal.symbol,
                            side=Side.SELL,
                            size=size_to_sell,
                            order_type=OrderType.MARKET,
                            timestamp=signal.timestamp,
                            stop_loss=current_price * 1.02,
                            take_profit=current_price * 0.96,
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

    def _volatility_adjustment(self, symbol: str, data_handler: DataHandler) -> float:
        """Calculate allocation adjustment based on recent volatility.

        Normalizes risk so that position size is reduced in high-volatility regimes.
        """
        try:
            data = data_handler.get_data(symbol, lookback=50)
            if len(data) < 20:
                return 1.0

            # Annualized volatility of log returns
            returns = np.log(data["close"] / data["close"].shift(1)).dropna()
            current_vol = returns.std() * np.sqrt(8760)  # Annualized

            # Baseline target volatility (e.g. 40% annual)
            target_vol = 0.40

            adj = target_vol / (current_vol + 1e-10)
            return float(np.clip(adj, 0.2, 2.0))
        except:
            return 1.0

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
