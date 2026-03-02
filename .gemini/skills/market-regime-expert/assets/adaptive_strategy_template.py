"""Adaptive regime-based ML Strategy."""

import pandas as pd
from typing import List
from crypto_analysis.signals.strategy import MLStrategy, Order, Side, OrderType, PortfolioManager, DataHandler
from crypto_analysis.online.detection.regime import RegimeDetector
from crypto_analysis.signals.base import Signal, SignalType

class AdaptiveRegimeStrategy(MLStrategy):
    """Extends MLStrategy to adjust parameters based on market regimes."""

    def __init__(self, symbols: List[str], aggregator, regime_detector: RegimeDetector = None):
        super().__init__(symbols, aggregator)
        self.regime_detector = regime_detector or RegimeDetector()

    def generate_signals(self, data_handler: DataHandler, portfolio: PortfolioManager) -> List[Order]:
        """Override to update regime before signal generation."""
        orders = []
        for symbol in self.symbols:
            data = data_handler.get_data(symbol, lookback=200)
            if len(data) < 100: continue

            # Update regime for the current asset
            regime = self.regime_detector.update(data)
            
            # Handle Crash Regime immediately
            if regime.name == "crash":
                position = portfolio.get_position(symbol)
                if position and position.size != 0:
                    orders.append(Order(
                        symbol=symbol,
                        side=Side.SELL if position.size > 0 else Side.BUY,
                        size=abs(position.size),
                        order_type=OrderType.MARKET,
                        timestamp=pd.Timestamp.now(),
                        metadata={"reason": "CRASH_REGIME_PROTECTION"}
                    ))
                continue

            # Standard signal generation
            orders.extend(super().generate_signals(data_handler, portfolio))
            
        return orders

    def _signal_to_orders(self, signal: Signal, portfolio: PortfolioManager, data_handler: DataHandler) -> List[Order]:
        """Adjust Stop-Loss and sizing based on current regime."""
        orders = super()._signal_to_orders(signal, portfolio, data_handler)
        
        regime = self.regime_detector.current_regime
        if not regime: return orders

        # Adjust each generated order based on regime
        for order in orders:
            price = data_handler.get_current_price(order.symbol)
            
            if regime.name == "volatile":
                # Tighten SL and reduce size further
                order.stop_loss = price * 0.97 if order.side == Side.BUY else price * 1.03
                order.size *= 0.5 
            elif regime.name == "ranging":
                # Reduce target profit
                order.take_profit = price * 1.02 if order.side == Side.BUY else price * 0.98
                order.size *= 0.7

        return orders
