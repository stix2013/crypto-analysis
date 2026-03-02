#!/usr/bin/env python
"""Online learning training script for Binance futures.

Fetches historical data from Binance and trains online learning models
to predict price direction for backtesting.

Outputs:
    - signals_<symbol>.csv: Trading signals
    - model_<symbol>.joblib: Trained model (can be loaded for inference)
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from crypto_analysis.data import create_client
from crypto_analysis.online.generator import OnlineSignalGenerator

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train online learning model on crypto futures"
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=os.environ.get("TRAIN_SYMBOL", "ETHUSDT"),
        help="Trading pair symbol (e.g., ETHUSDT, BTCUSDT)",
    )
    parser.add_argument(
        "--interval",
        default=os.environ.get("TRAIN_INTERVAL", "15m"),
        help="Kline interval",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=int(os.environ.get("TRAIN_BARS", 5000)),
        help="Number of historical bars",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=int(os.environ.get("TRAIN_WARMUP_BARS", 1000)),
        help="Number of bars for initial training",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output CSV file for signals"
    )
    parser.add_argument(
        "--model-output", type=str, default=None, help="Output file for trained model"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=int(os.environ.get("TRAIN_SEQUENCE_LENGTH", 60)),
        help="Sequence length for LSTM",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"signals_{args.symbol.lower()}.csv"
    if args.model_output is None:
        args.model_output = f"model_{args.symbol.lower()}.joblib"

    print("=" * 60)
    print(f"Online Learning Training - {args.symbol} {args.interval}")
    print("=" * 60)

    client = create_client()

    print(f"\n[1/4] Fetching {args.bars} bars of historical data...")
    data = client.fetch_historical(args.symbol, args.interval, args.bars)
    print(f"Fetched {len(data)} candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    print(f"\n[2/4] Initial training on first {args.warmup_bars} bars...")
    warmup_data = data.iloc[: args.warmup_bars]

    generator = OnlineSignalGenerator(
        name=f"Online_{args.symbol}",
        sequence_length=args.sequence_length,
        update_frequency=10,
    )

    try:
        generator.fit(warmup_data)
    except Exception as e:
        print(f"Initial training failed: {e}")
        print("Attempting with more data...")
        warmup_data = data.iloc[: min(args.warmup_bars * 2, len(data) - 100)]
        generator.fit(warmup_data)

    print("\n[3/4] Running online learning simulation...")

    signals = []
    online_data = data.iloc[args.warmup_bars :]

    for i, idx in enumerate(online_data.index):
        lookback = data.loc[:idx]
        if len(lookback) < generator.lookback_period:
            continue

        signal_list = generator.generate(lookback)
        if signal_list:
            for sig in signal_list:
                signals.append(
                    {
                        "timestamp": sig.timestamp,
                        "symbol": args.symbol,
                        "signal_type": sig.signal_type.name,
                        "confidence": sig.confidence,
                        "prediction": sig.metadata.get("ensemble_prediction", 0),
                        "regime": sig.metadata.get("regime", "unknown"),
                    }
                )

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(online_data)} bars...")

    print("\n[4/4] Results:")
    print(f"  Total signals generated: {len(signals)}")

    if signals:
        signals_df = pd.DataFrame(signals)
        signal_counts = signals_df["signal_type"].value_counts()
        print("  Signal breakdown:")
        for sig_type, count in signal_counts.items():
            print(f"    {sig_type}: {count}")

        output_path = Path(args.output)
        signals_df.to_csv(output_path, index=False)
        print(f"\n  Signals saved to: {output_path}")

        print("\n  Recent signals:")
        print(signals_df.tail(10).to_string(index=False))
    else:
        print("  No signals generated - model may need more training data")

    model_path = Path(args.model_output)
    joblib.dump(generator, model_path)
    print(f"\n  Model saved to: {model_path}")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
