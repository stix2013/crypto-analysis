"""Trading performance analytics and metrics."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class PerformanceAnalyzer:
    """Calculates and reports trading performance metrics."""

    @staticmethod
    def calculate_metrics(
        equity_history: pd.Series,
        orders: List[Any],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 8760,  # Default to 1h bars
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.

        Args:
            equity_history: Series of equity values indexed by timestamp
            orders: List of executed orders
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of bars per year

        Returns:
            Dictionary of metrics
        """
        returns = equity_history.pct_change().dropna()

        if len(returns) == 0:
            return {}

        # Basic returns
        total_return = (equity_history.iloc[-1] / equity_history.iloc[0]) - 1

        # Annualized return (CAGR-like for the period)
        num_years = len(equity_history) / periods_per_year
        if num_years > 0:
            annual_return = (1 + total_return) ** (1 / num_years) - 1
        else:
            annual_return = 0.0

        # Volatility
        annual_vol = returns.std() * np.sqrt(periods_per_year)

        # Sharpe Ratio
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period
        sharpe = (excess_returns.mean() / (returns.std() + 1e-10)) * np.sqrt(periods_per_year)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-10
        sortino = (excess_returns.mean() / (downside_std + 1e-10)) * np.sqrt(periods_per_year)

        # Drawdown
        rolling_max = equity_history.cummax()
        drawdowns = (equity_history - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Trade metrics
        trade_metrics = PerformanceAnalyzer._calculate_trade_metrics(orders)

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            **trade_metrics,
        }

        return metrics

    @staticmethod
    def _calculate_trade_metrics(orders: List[Any]) -> Dict[str, Any]:
        """Calculate metrics related to individual trades."""
        # This is a simplification; a "trade" usually spans multiple orders
        # For now, we'll just use the number of orders as a proxy
        if not orders:
            return {"num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0}

        # To calculate real win rate, we'd need to pair entries and exits
        # For a simplified version, we return basic counts
        return {"num_trades": len(orders), "num_symbols": len(set(o.symbol for o in orders))}

    @staticmethod
    def plot_equity_curve(equity_history: pd.DataFrame, title: str = "Equity Curve") -> Any:
        """Generate a plot of the equity curve and drawdowns.

        Note: Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Equity Curve
            ax1.plot(equity_history.index, equity_history["equity"], label="Portfolio Equity")
            ax1.set_title(title)
            ax1.set_ylabel("Value ($)")
            ax1.grid(True)
            ax1.legend()

            # Drawdown
            rolling_max = equity_history["equity"].cummax()
            drawdown = (equity_history["equity"] - rolling_max) / rolling_max
            ax2.fill_between(equity_history.index, drawdown, 0, color="red", alpha=0.3)
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Time")
            ax2.grid(True)

            plt.tight_layout()
            return fig
        except ImportError:
            print("matplotlib not found. Plotting disabled.")
            return None
