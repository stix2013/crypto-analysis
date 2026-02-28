#!/usr/bin/env python
"""Inference script for trained online learning model.

Loads a trained model and generates trading signals for new data from Binance.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from crypto_analysis.data import create_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate signals with trained model")
    parser.add_argument("model", help="Path to trained model (.joblib file)")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading pair symbol")
    parser.add_argument("--interval", default="15m", help="Kline interval")
    parser.add_argument("--bars", type=int, default=200, help="Number of recent bars")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file for signals")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    print("=" * 60)
    print(f"Inference - {args.symbol} {args.interval}")
    print("=" * 60)

    print(f"\n[1/3] Loading model from {model_path}...")
    generator = joblib.load(model_path)
    print(f"  Model loaded: {generator.name}")

    client = create_client()

    print(f"\n[2/3] Fetching {args.bars} recent bars...")
    data = client.fetch_historical(args.symbol, args.interval, args.bars)
    print(f"  Fetched {len(data)} candles")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")

    print("\n[3/3] Generating signals...")
    signals = generator.generate(data)

    print(f"\n  Signals generated: {len(signals)}")

    if signals:
        signal_data = []
        for sig in signals:
            signal_data.append(
                {
                    "timestamp": sig.timestamp,
                    "symbol": sig.symbol,
                    "signal_type": sig.signal_type.name,
                    "confidence": sig.confidence,
                    "prediction": sig.metadata.get("ensemble_prediction", 0),
                    "regime": sig.metadata.get("regime", "unknown"),
                }
            )

        signals_df = pd.DataFrame(signal_data)
        signal_counts = signals_df["signal_type"].value_counts()
        print("  Signal breakdown:")
        for sig_type, count in signal_counts.items():
            print(f"    {sig_type}: {count}")

        if args.output:
            output_path = Path(args.output)
            signals_df.to_csv(output_path, index=False)
            print(f"\n  Signals saved to: {output_path}")

        print("\n  Signals:")
        print(signals_df.to_string(index=False))
    else:
        # Show debug info when no signals
        import numpy as np

        features_df = generator.get_features(data)
        if len(features_df) > 0:
            scaled = generator.scaler.transform(features_df.values)
            current_point = scaled[-1:].reshape(1, -1)
            pred_rf = generator.rf.predict(current_point)[0]
            pred_pa = generator.pa_classifier.predict(current_point)[0]
            regime = generator.regime_detector.current_regime
            threshold = generator._get_regime_threshold(regime) if regime else 0.1

            print("  No signals (below threshold)")
            print(f"    Regime: {regime.name if regime else 'unknown'}")
            print(f"    Threshold: {threshold}")
            print(f"    RF prediction: {pred_rf}")
            print(f"    PA prediction: {pred_pa}")
            print(f"    Ensemble: {(pred_rf * 0.25 + (pred_pa * 2 - 1) * 0.25) / 1.0}")

    print(f"\n{'=' * 60}")
    print("Inference complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
