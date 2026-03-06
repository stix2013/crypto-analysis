"""Backtesting engine for cryptocurrency trading strategies."""

from typing import Any

import pandas as pd
from tqdm import tqdm

from crypto_analysis.signals.strategy import (
    DataHandler,
    Order,
    OrderType,
    PortfolioManager,
    Side,
    Strategy,
)
from crypto_analysis.utils.analytics import PerformanceAnalyzer


class Backtester:
    """Backtesting engine for cryptocurrency trading strategies.

    Supports both event-driven strategy testing and simple signal-based processing.
    """

    def __init__(
        self,
        strategy: Strategy | None = None,
        data: dict[str, pd.DataFrame] | None = None,
        initial_equity: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_pct: float = 0.0005,
        **kwargs: Any,
    ) -> None:
        """Initialize backtester.

        Args:
            strategy: Trading strategy to test (optional for signal-based)
            data: Dictionary mapping symbols to OHLCV DataFrames (optional)
            initial_equity: Starting capital
            commission_rate: Trading commission rate
            slippage_pct: Expected slippage percentage
            **kwargs: Additional arguments for backward compatibility
        """
        self.strategy = strategy
        self.data = data or {}
        self.data_handler = DataHandler()

        # Handle backward compatibility for initial_capital
        equity = kwargs.get("initial_capital", initial_equity)
        # Handle backward compatibility for commission
        comm = kwargs.get("commission", commission_rate)

        self.portfolio = PortfolioManager(
            initial_equity=equity,
            commission_rate=comm,
            slippage_pct=slippage_pct,
        )
        self.equity_history: list[dict[str, Any]] = []

    def process_signal(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        signal_type: str,
        price: float,
    ) -> None:
        """Process a single trade signal.

        Args:
            timestamp: Signal timestamp
            symbol: Trading pair symbol
            signal_type: Signal type ('BUY', 'SELL', 'EXIT', 'CLOSE')
            price: Execution price
        """
        # 1. Update data handler with current price
        temp_df = pd.DataFrame(
            {"close": [price]},
            index=[timestamp],
        )
        self.data_handler.load_data(symbol, temp_df)

        # 2. Determine target position size
        target_size = 0.0
        if signal_type.upper() == "BUY":
            target_size = 1.0
        elif signal_type.upper() == "SELL":
            target_size = -1.0
        elif signal_type.upper() in ["EXIT", "CLOSE"]:
            target_size = 0.0
        else:
            # Ignore unknown signal types
            return

        # 3. Calculate order needed to reach target size
        pos = self.portfolio.get_position(symbol)
        current_size = pos.size if pos else 0.0

        if current_size != target_size:
            needed_size = target_size - current_size
            order_side = Side.BUY if needed_size > 0 else Side.SELL

            order = Order(
                symbol=symbol,
                side=order_side,
                size=abs(needed_size),
                order_type=OrderType.MARKET,
                timestamp=timestamp,
            )
            self.portfolio.execute_order(order, self.data_handler)

        # 4. Record equity
        total_equity = self.portfolio.get_total_equity(self.data_handler)
        self.equity_history.append(
            {
                "timestamp": timestamp,
                "equity": total_equity,
                "cash": self.portfolio.cash,
                "positions": {s: p.size for s, p in self.portfolio.positions.items()},
            }
        )

    def get_equity_curve(self) -> pd.Series:
        """Get the equity curve as a pandas Series."""
        if not self.equity_history:
            return pd.Series()
        df = pd.DataFrame(self.equity_history).set_index("timestamp")
        return df["equity"]

    def get_trades(self) -> pd.DataFrame:
        """Get completed trades as a DataFrame."""
        columns = ["timestamp", "symbol", "side", "size", "price", "pnl"]
        trades = []
        for order in self.portfolio.orders:
            trades.append(
                {
                    "timestamp": order.timestamp,
                    "symbol": order.symbol,
                    "side": order.side.name,
                    "size": order.size,
                    "price": order.price,
                    "pnl": order.metadata.get("pnl", 0.0),
                }
            )
        return pd.DataFrame(trades, columns=columns)

    def run(self, start_idx: int = 100) -> dict[str, Any]:
        """Run the backtest simulation.

        Args:
            start_idx: Starting index in data to begin trading

        Returns:
            Dictionary with backtest results and metrics

        Raises:
            ValueError: If strategy or data not provided
        """
        if self.strategy is None:
            raise ValueError("Strategy must be provided for event-driven backtesting.")
        if not self.data:
            raise ValueError("Data must be provided for event-driven backtesting.")

        # Get common index (assuming aligned data or using first symbol)
        first_symbol = self.strategy.symbols[0]
        full_df = self.data[first_symbol]
        indices = full_df.index[start_idx:]

        print(f"Running backtest from {indices[0]} to {indices[-1]}...")

        for _i, timestamp in enumerate(tqdm(indices)):
            # 1. Update data handler with data up to current timestamp
            for symbol in self.strategy.symbols:
                current_view = self.data[symbol].loc[:timestamp]
                self.data_handler.load_data(symbol, current_view)

            # 2. Check for SL/TP triggers BEFORE generating new signals
            self.portfolio.check_risk_triggers(self.data_handler)

            # 3. Generate signals from strategy
            orders = self.strategy.generate_signals(self.data_handler, self.portfolio)

            # 3. Execute orders
            for order in orders:
                self.portfolio.execute_order(order, self.data_handler)

            # 4. Record daily equity
            total_equity = self.portfolio.get_total_equity(self.data_handler)
            self.equity_history.append(
                {
                    "timestamp": timestamp,
                    "equity": total_equity,
                    "cash": self.portfolio.cash,
                    "positions": {
                        s: p.size for s, p in self.portfolio.positions.items()
                    },
                }
            )

        return self._calculate_results()

    def _calculate_results(self) -> dict[str, Any]:
        """Calculate performance metrics from equity history."""
        if not self.equity_history:
            return {}

        history_df = pd.DataFrame(self.equity_history).set_index("timestamp")

        metrics = PerformanceAnalyzer.calculate_metrics(
            history_df["equity"], self.portfolio.orders
        )

        results = {
            **metrics,
            "final_equity": history_df["equity"].iloc[-1],
            "equity_history": history_df,
        }

        return results
