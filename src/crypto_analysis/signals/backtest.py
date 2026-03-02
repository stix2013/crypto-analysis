"""Backtesting engine for cryptocurrency trading strategies."""

import pandas as pd
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from crypto_analysis.signals.strategy import (
    Strategy,
    DataHandler,
    PortfolioManager,
    Order,
    Side,
    OrderType,
)
from crypto_analysis.utils.analytics import PerformanceAnalyzer


class Backtester:
    """Event-driven backtesting engine.

    Simulates trading by iterating through historical data and
    executing strategy decisions.
    """

    def __init__(
        self,
        strategy: Strategy,
        data: Dict[str, pd.DataFrame],
        initial_equity: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        """Initialize backtester.

        Args:
            strategy: Trading strategy to test
            data: Dictionary mapping symbols to OHLCV DataFrames
            initial_equity: Starting capital
            commission_rate: Trading commission rate
            slippage_pct: Expected slippage percentage
        """
        self.strategy = strategy
        self.data = data
        self.data_handler = DataHandler()
        self.portfolio = PortfolioManager(
            initial_equity=initial_equity,
            commission_rate=commission_rate,
            slippage_pct=slippage_pct,
        )
        self.equity_history: List[Dict[str, Any]] = []

    def run(self, start_idx: int = 100) -> Dict[str, Any]:
        """Run the backtest simulation.

        Args:
            start_idx: Starting index in data to begin trading

        Returns:
            Dictionary with backtest results and metrics
        """
        # Get common index (assuming aligned data or using first symbol)
        first_symbol = self.strategy.symbols[0]
        full_df = self.data[first_symbol]
        indices = full_df.index[start_idx:]

        print(f"Running backtest from {indices[0]} to {indices[-1]}...")

        for i, timestamp in enumerate(tqdm(indices)):
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
                    "positions": {s: p.size for s, p in self.portfolio.positions.items()},
                }
            )

        return self._calculate_results()

    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate performance metrics from equity history."""
        if not self.equity_history:
            return {}

        history_df = pd.DataFrame(self.equity_history).set_index("timestamp")

        metrics = PerformanceAnalyzer.calculate_metrics(history_df["equity"], self.portfolio.orders)

        results = {
            **metrics,
            "final_equity": history_df["equity"].iloc[-1],
            "equity_history": history_df,
        }

        return results
