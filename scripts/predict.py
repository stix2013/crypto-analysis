#!/usr/bin/env python
"""Inference script for trained online learning model.

Loads a trained model and generates trading signals for new data from Binance.
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from crypto_analysis.data import create_client

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate signals with trained model")
    parser.add_argument(
        "model",
        nargs="?",
        default=os.environ.get("PREDICT_MODEL", "model_ethusdt.joblib"),
        help="Path to trained model (.joblib file)",
    )
    parser.add_argument(
        "--symbol",
        default=os.environ.get("PREDICT_SYMBOL", "ETHUSDT"),
        help="Trading pair symbol",
    )
    parser.add_argument(
        "--interval",
        default=os.environ.get("PREDICT_INTERVAL", "15m"),
        help="Kline interval",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=int(os.environ.get("PREDICT_BARS", 200)),
        help="Number of recent bars",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output CSV file for signals"
    )
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
    print(f"  Required lookback: {generator.lookback_period} bars")

    # Ensure we fetch enough bars for both feature engineering (200) and model sequence
    # FeatureEngineer uses ma_200, so we need at least 200 bars just for features
    feature_lookback = 200
    fetch_bars = max(args.bars, feature_lookback + generator.sequence_length + 10)

    client = create_client()

    print(f"\n[2/3] Fetching {fetch_bars} recent bars...")
    data = client.fetch_historical(args.symbol, args.interval, fetch_bars)
    # Add symbol name to data for the generator
    data.index.name = args.symbol
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

        features_df = generator.get_features(data)
        if len(features_df) > 0:
            # We need to reconstruct the ensemble prediction for debugging
            scaled = generator.scaler.transform(features_df.values)
            current_point = scaled[-1:].reshape(1, -1)

            debug_preds = {}
            if generator.lstm is not None and len(scaled) >= generator.sequence_length:
                try:
                    current_seq = scaled[-generator.sequence_length :].reshape(
                        1, generator.sequence_length, -1
                    )
                    debug_preds["lstm"] = generator.lstm.predict(current_seq)[0][0]
                except:
                    pass

            if generator.nn is not None:
                try:
                    debug_preds["nn"] = generator.nn.predict(current_point)[0] * 2 - 1
                except:
                    pass

            try:
                debug_preds["rf"] = generator.rf.predict(current_point)[0] * 2 - 1
            except:
                pass

            try:
                debug_preds["pa"] = (
                    generator.pa_classifier.predict(current_point)[0] * 2 - 1
                )
            except:
                pass

            regime = generator.regime_detector.current_regime
            threshold = generator._get_regime_threshold(regime) if regime else 0.1

            available_weights = {m: generator.model_weights[m] for m in debug_preds}
            total_w = sum(available_weights.values()) or 1.0
            ensemble = sum(
                debug_preds[m] * (available_weights[m] / total_w) for m in debug_preds
            )

            print("  No signals (below threshold)")
            print(f"    Regime: {regime.name if regime else 'unknown'}")
            print(f"    Threshold: {threshold:.4f}")
            print(f"    Ensemble: {ensemble:.4f}")
            print("    Individual predictions:")
            for m, p in debug_preds.items():
                w = available_weights[m] / total_w
                print(f"      {m:4}: {p:7.4f} (weight: {w:.2f})")

    print(f"\n{'=' * 60}")
    print("Inference complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
