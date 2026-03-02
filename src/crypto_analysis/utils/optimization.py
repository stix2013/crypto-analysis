"""Strategy parameter optimization and grid search."""

import itertools
from typing import Dict, List, Any, Callable
import pandas as pd
from tqdm import tqdm

from crypto_analysis.signals.backtest import Backtester


class ParameterOptimizer:
    """Optimizes strategy parameters using grid search."""

    def __init__(
        self,
        strategy_factory: Callable[[List[str], Dict[str, Any]], Any],
        data: Dict[str, pd.DataFrame],
        symbols: List[str],
        initial_equity: float = 10000.0,
    ) -> None:
        """Initialize optimizer.

        Args:
            strategy_factory: Function that creates a strategy from symbols and params
            data: Market data
            symbols: Symbols to trade
            initial_equity: Starting capital
        """
        self.strategy_factory = strategy_factory
        self.data = data
        self.symbols = symbols
        self.initial_equity = initial_equity
        self.results: List[Dict[str, Any]] = []

    def grid_search(self, param_grid: Dict[str, List[Any]]) -> pd.DataFrame:
        """Run grid search over parameter combinations.

        Args:
            param_grid: Dictionary mapping param names to lists of values

        Returns:
            DataFrame with results for all combinations
        """
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))

        print(f"Starting grid search with {len(combinations)} combinations...")

        for values in tqdm(combinations):
            params = dict(zip(keys, values))

            # Create strategy with these params
            strategy = self.strategy_factory(self.symbols, params)

            # Run backtest
            backtester = Backtester(
                strategy=strategy, data=self.data, initial_equity=self.initial_equity
            )

            try:
                metrics = backtester.run(start_idx=100)

                # Store results
                result = {
                    **params,
                    "total_return": metrics.get("total_return", 0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "num_trades": metrics.get("num_trades", 0),
                    "final_equity": metrics.get("final_equity", self.initial_equity),
                }
                self.results.append(result)
            except Exception as e:
                print(f"Error for params {params}: {e}")

        return pd.DataFrame(self.results).sort_values("sharpe_ratio", ascending=False)
