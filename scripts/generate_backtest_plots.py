#!/usr/bin/env python
"""Generate visualization plots from backtest results.

Usage:
    python scripts/generate_backtest_plots.py signals/btcusdt_1h_signals.csv
    python scripts/generate_backtest_plots.py signals/btcusdt_1h_signals.csv --equity signals/equity_history.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from crypto_analysis.data import create_client
from crypto_analysis.settings import get_settings
from crypto_analysis.signals.backtest import Backtester

# Get initial capital from settings
_settings = get_settings()
DEFAULT_INITIAL_CAPITAL = _settings.backtest.initial_capital


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from backtest results"
    )
    parser.add_argument(
        "signals_path",
        type=str,
        help="Path to signals CSV file",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Kline interval (e.g., 1h, 15m)",
    )
    parser.add_argument(
        "--equity",
        type=str,
        default=None,
        help="Path to equity history CSV (optional)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help=f"Initial capital for backtest (default: from settings or {int(DEFAULT_INITIAL_CAPITAL)})",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0004,
        help="Commission rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="signals",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    signals_file = Path(args.signals_path)
    if not signals_file.exists():
        print(f"Error: Signals file not found: {signals_file}")
        return

    signals_df = pd.read_csv(signals_file)
    if signals_df.empty:
        print("Error: Signals file is empty")
        return

    signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])

    # Auto-detect symbol from signals file if not provided
    if "symbol" in signals_df.columns:
        detected_symbol = signals_df["symbol"].iloc[0]
        if args.symbol == "BTCUSDT":  # Only override if still default
            args.symbol = detected_symbol

    # Auto-detect interval from timestamp differences
    if len(signals_df) >= 2:
        time_diffs = signals_df["timestamp"].diff().dropna()
        if not time_diffs.empty:
            median_diff = time_diffs.median()
            minutes = int(median_diff.total_seconds() / 60)
            interval_map = {
                1: "1m",
                5: "5m",
                15: "15m",
                30: "30m",
                60: "1h",
                240: "4h",
                1440: "1d",
            }
            if (
                args.interval == "1h" and minutes in interval_map
            ):  # Only override if still default
                args.interval = interval_map[minutes]

    print(f"Loaded {len(signals_df)} signals from {signals_file}")
    print(f"Running backtest for {args.symbol} {args.interval}...")

    client = create_client()
    signal_start = signals_df["timestamp"].min()
    signal_end = signals_df["timestamp"].max()
    time_span_hours = (signal_end - signal_start).total_seconds() / 3600

    # Convert interval to minutes for bar calculation
    interval_mins = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }.get(args.interval, 60)
    bars_needed = int(time_span_hours * 60 / interval_mins) + 500

    print(
        f"  Signals span: {time_span_hours:.0f}h, fetching {bars_needed} bars from {signal_start}..."
    )

    # Fetch data that covers the signal time range
    price_data = client.fetch_historical(
        args.symbol, args.interval, bars_needed, start_time=signal_start
    )
    price_data_sorted = price_data.sort_index().tz_localize(None)

    backtester = Backtester(
        initial_capital=args.initial_capital,
        commission=args.commission,
        generate_plots=True,
        output_dir=args.output_dir,
    )
    backtester.set_price_data(price_data)

    time_span_hours = (signal_end - signal_start).total_seconds() / 3600

    # Convert interval to minutes for bar calculation
    interval_mins = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }.get(args.interval, 60)
    bars_needed = int(time_span_hours * 60 / interval_mins) + 500

    print(
        f"  Signals span: {time_span_hours:.0f}h, fetching {bars_needed} bars from {signal_start}..."
    )

    backtester = Backtester(
        initial_capital=args.initial_capital,
        commission=args.commission,
        generate_plots=True,
        output_dir=args.output_dir,
    )
    backtester.set_price_data(price_data)

    price_data_sorted = price_data.sort_index().tz_localize(None)
    processed = 0

    for _, row in signals_df.iterrows():
        timestamp = pd.to_datetime(row["timestamp"])
        signal_type = row["signal_type"]

        timestamp_naive = timestamp.tz_localize(None)

        if timestamp_naive < price_data_sorted.index[0]:
            continue
        if timestamp_naive > price_data_sorted.index[-1]:
            continue

        nearest_idx = price_data_sorted.index.get_indexer(
            [timestamp_naive], method="nearest"
        )[0]
        nearest_ts = price_data_sorted.index[nearest_idx]
        current_price = price_data_sorted.loc[nearest_ts, "close"]

        backtester.process_signal(
            timestamp=nearest_ts,
            symbol=args.symbol,
            signal_type=signal_type,
            price=current_price,
        )
        processed += 1

    print(f"  Processed {processed}/{len(signals_df)} signals")

    history_df = pd.DataFrame(backtester.equity_history).set_index("timestamp")
    backtester._generate_visualization(history_df)

    equity_curve = backtester.get_equity_curve()
    trades = backtester.get_trades()

    print("\nBacktest Results:")
    print(f"  Initial Capital: ${args.initial_capital:,.2f}")

    if len(equity_curve) == 0:
        print(
            "  ERROR: No equity data generated - signals may not have matched price data"
        )
        print(f"  Price data range: {price_data.index[0]} to {price_data.index[-1]}")
        print(
            f"  Signals range:   {signals_df['timestamp'].min()} to {signals_df['timestamp'].max()}"
        )
        return

    print(f"  Final Equity:    ${equity_curve.iloc[-1]:,.2f}")
    print(f"  Total Trades:   {len(trades)}")
    if len(trades) > 0:
        print(f"  Winning Trades: {(trades['pnl'] > 0).sum()}")
        print(f"  Total PnL:      ${trades['pnl'].sum():,.2f}")

    print(f"\nPlots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
