from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from celery import shared_task
from crypto_analysis.data import create_client
from crypto_analysis.online.generator import OnlineSignalGenerator
from crypto_analysis.signals.backtest import Backtester
from dotenv import load_dotenv

load_dotenv()


@shared_task(name="fetch_market_data")
def fetch_market_data(
    symbol: str,
    interval: str,
    bars: int,
) -> dict[str, Any]:
    client = create_client()
    data = client.fetch_historical(symbol, interval, bars)  # type: ignore[arg-type]
    return {
        "symbol": symbol,
        "interval": interval,
        "bars": len(data),
        "start_time": str(data.index[0]),
        "end_time": str(data.index[-1]),
    }


@shared_task(name="train_model")
def train_model(
    symbol: str,
    interval: str = "1h",
    bars: int = 5000,
    warmup_bars: int = 1000,
    sequence_length: int = 60,
    output_dir: str = "/app/signals",
    model_dir: str = "/app/models",
) -> dict[str, Any]:
    client = create_client()
    data = client.fetch_historical(symbol, interval, bars)  # type: ignore[arg-type]

    generator = OnlineSignalGenerator(
        name=f"Online_{symbol}",
        sequence_length=sequence_length,
        update_frequency=10,
    )

    warmup_data = data.iloc[:warmup_bars]
    generator.fit(warmup_data)

    signals: list[dict[str, Any]] = []
    online_data = data.iloc[warmup_bars:]

    # Pre-calculate features once for the entire dataset
    # This avoids O(N^2) complexity in the loop
    all_features = generator.feature_engineer.create_features(data)

    for _i, idx in enumerate(online_data.index):
        lookback = data.loc[:idx]
        if len(lookback) < generator.lookback_period:
            continue

        # Pass the pre-calculated features for this lookback period
        current_features = all_features.loc[:idx]
        signal_list = generator.generate(lookback, features_df=current_features)

        if signal_list:
            for sig in signal_list:
                signals.append(
                    {
                        "timestamp": sig.timestamp,
                        "symbol": symbol,
                        "signal_type": sig.signal_type.name,
                        "confidence": sig.confidence,
                        "prediction": sig.metadata.get("ensemble_prediction", 0),
                        "regime": sig.metadata.get("regime", "unknown"),
                    }
                )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    columns = [
        "timestamp",
        "symbol",
        "signal_type",
        "confidence",
        "prediction",
        "regime",
    ]
    signals_df = pd.DataFrame(signals, columns=columns)
    signals_file = output_path / f"signals_{symbol.lower()}_{interval.lower()}.csv"
    signals_df.to_csv(signals_file, index=False)

    model_output_path = Path(model_dir)
    model_output_path.mkdir(parents=True, exist_ok=True)
    model_file = model_output_path / f"model_{symbol.lower()}_{interval.lower()}.joblib"

    joblib.dump(generator, model_file)

    return {
        "symbol": symbol,
        "interval": interval,
        "bars": len(data),
        "start_time": str(data.index[0]),
        "end_time": str(data.index[-1]),
        "signals_file": str(signals_file),
        "model_file": str(model_file),
        "total_signals": len(signals),
    }


@shared_task(name="run_prediction")
def run_prediction(
    model_path: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    bars: int = 200,
) -> dict[str, Any]:
    model_file = Path(model_path)

    # If the path doesn't exist, try to construct it from symbol and interval in the default directory
    if not model_file.exists():
        default_path = (
            Path("/app/models") / f"model_{symbol.lower()}_{interval.lower()}.joblib"
        )
        if default_path.exists():
            model_file = default_path
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path} or {default_path}"
            )

    generator = joblib.load(model_file)

    feature_lookback = 200
    fetch_bars = max(bars, feature_lookback + generator.sequence_length + 10)

    client = create_client()
    data = client.fetch_historical(symbol, interval, fetch_bars)  # type: ignore[arg-type]

    signals = generator.generate(data)

    result: dict[str, Any] = {
        "symbol": symbol,
        "signals_count": len(signals),
        "signals": [],
    }

    for sig in signals:
        result["signals"].append(
            {
                "timestamp": str(sig.timestamp),
                "signal_type": sig.signal_type.name,
                "confidence": sig.confidence,
                "prediction": sig.metadata.get("ensemble_prediction", 0),
                "regime": sig.metadata.get("regime", "unknown"),
            }
        )

    return result


@shared_task(name="run_backtest")
def run_backtest(
    signals_path: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    initial_capital: float = 10000.0,
    commission: float = 0.0004,
) -> dict[str, Any]:
    signals_file = Path(signals_path)
    if not signals_file.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_path}")

    signals_df = pd.read_csv(signals_file)
    if signals_df.empty:
        return {
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "message": "No signals generated for backtest",
        }

    signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])

    client = create_client()
    bars_needed = len(signals_df) + 200
    price_data = client.fetch_historical(symbol, interval, bars_needed)  # type: ignore[arg-type]

    backtester = Backtester(
        initial_capital=initial_capital,
        commission=commission,
    )

    for _, row in signals_df.iterrows():
        timestamp = row["timestamp"]
        signal_type = row["signal_type"]

        if timestamp in price_data.index:
            current_price = price_data.loc[timestamp, "close"]
            backtester.process_signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                price=current_price,
            )

    equity_curve = backtester.get_equity_curve()
    trades = backtester.get_trades()

    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_equity": float(equity_curve.iloc[-1])
        if len(equity_curve) > 0
        else initial_capital,
        "total_trades": len(trades),
        "winning_trades": int((trades["pnl"] > 0).sum()) if len(trades) > 0 else 0,
        "total_pnl": float(trades["pnl"].sum()) if len(trades) > 0 else 0.0,
    }


@shared_task(name="train_and_backtest")
def train_and_backtest(
    symbol: str,
    interval: str = "1h",
    bars: int = 5000,
    warmup_bars: int = 1000,
) -> dict[str, Any]:
    train_result = train_model(symbol, interval, bars, warmup_bars)
    backtest_result = run_backtest(train_result["signals_file"], symbol, interval)
    return {"train": train_result, "backtest": backtest_result}
