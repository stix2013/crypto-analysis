"""Binance Futures data fetching for online learning."""

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import requests

from crypto_analysis.settings import get_settings


@dataclass
class BinanceConfig:
    """Configuration for Binance API connection."""

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://fapi.binance.com"
    testnet: bool = False

    def __post_init__(self) -> None:
        settings = get_settings()
        if not self.api_key:
            object.__setattr__(self, "api_key", settings.binance.api_key)
        if not self.secret_key:
            object.__setattr__(self, "secret_key", settings.binance.secret_key)


Interval = Literal[
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
]


class BinanceClient:
    """Client for fetching Binance USDT-M Futures OHLCV data.

    Uses direct HTTP requests to Binance Futures API.

    Attributes:
        config: Binance API configuration
        session: requests Session for connection pooling
    """

    def __init__(self, config: BinanceConfig | None = None) -> None:
        """Initialize Binance client.

        Args:
            config: Optional configuration, loads from env if not provided
        """
        self.config = config or BinanceConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "crypto-analysis/1.0"})

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests.

        Args:
            query_string: Query parameters as string

        Returns:
            HMAC SHA256 signature
        """
        return hmac.new(
            self.config.secret_key.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        signed: bool = False,
    ) -> list[list] | dict[str, str] | dict[str, float]:
        """Make HTTP request to Binance API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            signed: Whether request needs authentication

        Returns:
            Response JSON

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.config.base_url}{endpoint}"
        params = params or {}

        if signed and self.config.api_key and self.config.secret_key:
            params["timestamp"] = int(time.time() * 1000)
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            params["signature"] = self._generate_signature(query_string)
            self.session.headers.update({"X-MBX-APIKEY": self.config.api_key})

        if method == "GET":
            response = self.session.get(url, params=params)
        elif method == "POST":
            response = self.session.post(url, params=params)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: Interval,
        start_str: str | None = None,
        end_str: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV (candlestick) data from Binance Futures.

        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            interval: Kline interval (e.g., '15m', '1h')
            start_str: Start time as ISO string or Unix timestamp (ms)
            end_str: End time as ISO string or Unix timestamp (ms)
            limit: Number of klines to fetch (max 1500)

        Returns:
            DataFrame with columns: open, high, low, close, volume

        Raises:
            requests.HTTPError: If API request fails
        """
        params: dict = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1500),
        }

        if start_str:
            params["startTime"] = self._parse_time(start_str)
        if end_str:
            params["endTime"] = self._parse_time(end_str)

        klines = self._request("GET", "/fapi/v1/klines", params)

        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.set_index("open_time")
        df = df[numeric_cols]

        return df

    def fetch_historical(
        self,
        symbol: str,
        interval: Interval,
        bars: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data efficiently.

        Uses a backward-fetching approach to get the most recent bars
        efficiently by fetching progressively older data.

        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            interval: Kline interval
            bars: Number of bars to fetch (default 1000)

        Returns:
            DataFrame with OHLCV data
        """
        bars = int(bars)
        # Initial fetch of latest data
        df = self.fetch_ohlcv(symbol, interval, limit=min(bars, 1500))

        # Progressively fetch older data if we need more
        while len(df) < bars:
            # Fetch data older than our current earliest bar
            first_time = int(df.index[0].timestamp() * 1000) - 1
            more_data = self._request(
                "GET",
                "/fapi/v1/klines",
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "endTime": first_time,
                    "limit": min(bars - len(df), 1500),
                },
            )

            if not more_data:
                break

            new_df = pd.DataFrame(
                more_data,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ],
            )

            new_df["open_time"] = pd.to_datetime(new_df["open_time"], unit="ms")
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

            new_df = new_df.set_index("open_time")
            new_df = new_df[numeric_cols]

            # Older data goes at the top
            df = pd.concat([new_df, df])
            df = df[~df.index.duplicated(keep="last")]

            # If we didn't get much data, might be at the beginning of history
            if len(more_data) < 10:
                break

        return df.tail(bars)

    def fetch_recent(
        self,
        symbol: str,
        interval: Interval,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch most recent OHLCV data.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            limit: Number of bars to fetch

        Returns:
            DataFrame with recent OHLCV data
        """
        return self.fetch_ohlcv(symbol, interval, limit=limit)

    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price
        """
        ticker = self._request(
            "GET",
            "/fapi/v1/ticker/24hr",
            {"symbol": symbol.upper()},
        )
        return float(ticker["lastPrice"])  # type: ignore[call-overload]

    def _parse_time(self, time_str: str | int) -> int:
        """Parse time string to milliseconds.

        Args:
            time_str: ISO string, Unix timestamp, or relative time

        Returns:
            Timestamp in milliseconds
        """
        if isinstance(time_str, int):
            return time_str

        if isinstance(time_str, str):
            if time_str.isdigit():
                return int(time_str)
            try:
                dt = pd.to_datetime(time_str)
                return int(dt.timestamp() * 1000)
            except Exception:
                pass

        raise ValueError(f"Invalid time format: {time_str}")


def create_client() -> BinanceClient:
    """Create a Binance client from environment variables.

    Returns:
        Configured BinanceClient instance
    """
    return BinanceClient()
