"""
CCXT Provider
Fetches cryptocurrency market data using the CCXT library.
"""

import ccxt
import pandas as pd
import time


class CCXTProvider:
    """Provides cryptocurrency market data via CCXT."""

    def __init__(self):
        self.supported_exchanges = {
            "binance": ccxt.binance,
            "kraken": ccxt.kraken,
            "coinbase": ccxt.coinbase,
            "kucoin": ccxt.kucoin,
            "bitfinex": ccxt.bitfinex,
        }

    def fetch_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        retries: int = 3,
    ):
        """
        Fetch OHLCV data from a cryptocurrency exchange.

        Args:
            exchange (str): Exchange name (e.g., 'binance').
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            timeframe (str): Candle timeframe.
            limit (int): Number of candles.
            retries (int): Number of retry attempts.

        Returns:
            pandas.DataFrame: OHLCV market data.
        """
        if exchange.lower() not in self.supported_exchanges:
            raise ValueError(f"Unsupported exchange: {exchange}")

        exchange_class = self.supported_exchanges[exchange.lower()]

        try:
            exchange_instance = exchange_class({
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {
                    "defaultType": "spot",  # Avoid derivatives endpoints
                },
            })

            print(f"[INFO] Fetching {symbol} from {exchange}...")

            for attempt in range(retries):
                try:
                    ohlcv = exchange_instance.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        limit=limit,
                    )

                    if not ohlcv:
                        raise ValueError("No data returned from exchange.")

                    df = pd.DataFrame(
                        ohlcv,
                        columns=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ],
                    )

                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], unit="ms"
                    )
                    df.sort_values("timestamp", inplace=True)
                    df.reset_index(drop=True, inplace=True)

                    return df

                except Exception as e:
                    print(
                        f"[WARNING] Attempt {attempt + 1} failed: {e}"
                    )
                    time.sleep(3)

            raise RuntimeError(
                f"Failed to fetch data from {exchange} for {symbol}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch data from {exchange} for {symbol}: {e}"
            )