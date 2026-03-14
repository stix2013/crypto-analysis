#!/usr/bin/env python
"""Inference script for trained online learning model.

Simple prediction script using the new prediction API.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from crypto_analysis.data import create_client
from crypto_analysis.signals.predict import Predictor, resolve_model_path
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate signals with trained model")
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Trading pair symbol (e.g., BTCUSDT). Uses PREDICT_SYMBOL env if not set.",
    )
    parser.add_argument(
        "--interval",
        "-i",
        default=None,
        help="Kline interval (e.g., 1h, 15m). Uses PREDICT_INTERVAL env if not set.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Explicit model path. Overrides symbol+interval.",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=int(os.environ.get("PREDICT_BARS", 500)),
        help="Number of recent bars to fetch",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV file for signals",
    )
    args = parser.parse_args()

    if args.output is None and args.symbol and args.interval:
        args.output = f"predict_{args.symbol.lower()}_{args.interval.lower()}.csv"

    print("=" * 60)
    print("Prediction - Trading Signal Generator")
    print("=" * 60)

    # Resolve model path
    try:
        model_path = resolve_model_path(
            symbol=args.symbol,
            model_path=args.model,
            interval=args.interval,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        return

    print(f"\nModel: {model_path}")

    # Generate predictions
    try:
        # If bars > 1, we might want to see multiple recent signals
        # For OnlineSignalGenerator, generate() only returns the signal for the last bar.
        # To get multiple signals, we need to simulate the sequence.

        predictor = Predictor(
            symbol=args.symbol,
            interval=args.interval,
            model_path=args.model,
        )

        client = create_client()
        data = client.fetch_historical(predictor.symbol, predictor.interval, args.bars)

        # Ensure symbol is attached for signal generator
        data.symbol = predictor.symbol

        if data.empty:
            print(f"No data found for {predictor.symbol}")
            return

        print(f"Fetched {len(data)} bars. Generating signals...")

        signals = []
        # We look at the last 50 bars by default for "recent" signals if bars is large
        # or all bars if bars is small.
        process_bars = min(len(data), 50)

        for i in range(len(data) - process_bars, len(data)):
            window = data.iloc[: i + 1]
            if len(window) < predictor.model.lookback_period:
                continue

            sig_list = predictor.model.generate(window)
            if sig_list:
                signals.extend(sig_list)

        # Always print the very latest prediction state
        # Need at least 200 (MA) + 60 (LSTM) = 260 bars. Use 300 for safety.
        feature_lookback = max(300, predictor.model.lookback_period)
        latest_window = data.tail(feature_lookback)
        if len(latest_window) >= feature_lookback:
            # We need to get the regime and prediction manually since generate()
            # might not have been called for the very last bar if it wasn't in signals
            regime = predictor.model.regime_detector.update(latest_window)

            # Get the actual computed ensemble prediction from the model's buffer
            avg_pred = 0.0
            if len(predictor.model.prediction_buffer) > 0:
                avg_pred = predictor.model.prediction_buffer[-1]["prediction"]
                print(f"  Symbol:    {predictor.symbol}")
                print(f"  Timestamp: {data.index[-1]}")
                print(f"  Regime:    {regime.name}")
                print(f"  Ensemble:  {avg_pred:.4f}")

                threshold = predictor.model._get_regime_threshold(regime)
                print(f"  Threshold: {threshold:.4f}")

                if abs(avg_pred) > threshold:
                    direction = "LONG" if avg_pred > 0 else "SHORT"
                    print(f"  Action:    ENTRY_{direction} (SIGNAL TRIGGERED)")
                else:
                    print("  Action:    WAIT (No signal)")
            else:
                print(
                    "\nWarning: Insufficient data to calculate features for the latest bar."
                )
                print(f"Required lookback: {feature_lookback} bars.")
        else:
            print(
                f"\nWarning: Not enough data for latest state report (needs {feature_lookback} bars, got {len(latest_window)})"
            )

    except Exception as e:
        print(f"\nPrediction failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"Signals generated: {len(signals)}")

    if signals:
        # Build signal DataFrame
        signal_data = []
        for sig in signals:
            # Fix symbol if it was extracted incorrectly from dataframe index
            display_symbol = sig.symbol
            if display_symbol == "open_time":
                display_symbol = predictor.symbol

            signal_data.append(
                {
                    "timestamp": sig.timestamp,
                    "symbol": display_symbol,
                    "type": sig.signal_type.name,
                    "conf": f"{sig.confidence:.3f}",
                    "pred": f"{sig.metadata.get('ensemble_prediction', 0):.4f}",
                    "regime": sig.metadata.get("regime", "unknown"),
                }
            )

        signals_df = pd.DataFrame(signal_data)

        # Print summary
        print("\nSignal breakdown:")
        counts = signals_df["type"].value_counts()
        for sig_type, count in counts.items():
            print(f"  {sig_type:15}: {count:3}")

        # Save to CSV if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            signals_df.to_csv(output_path, index=False)
            print(f"\nSignals saved to: {output_path}")

        # Print signals
        print("\nMost Recent Signals:")
        print("-" * 80)
        # Sort by timestamp to show most recent
        signals_df = signals_df.sort_values("timestamp", ascending=False)
        print(signals_df.head(20).to_string(index=False))
        if len(signals_df) > 20:
            print(f"... and {len(signals_df) - 20} more signals.")
        print("-" * 80)
    else:
        print("\nNo signals generated (predictions below threshold)")

    print(f"\n{'=' * 60}")
    print("Prediction complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
