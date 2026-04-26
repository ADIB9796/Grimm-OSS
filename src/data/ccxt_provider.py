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
        # Cache initialized exchanges to avoid recreating them in loops
        self._active_exchanges = {}

    def get_exchange(self, exchange_name: str):
        """Retrieves or initializes an exchange instance."""
        exchange_id = exchange_name.lower()
        if exchange_id not in self.supported_exchanges:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        if exchange_id not in self._active_exchanges:
            exchange_class = self.supported_exchanges[exchange_id]
            self._active_exchanges[exchange_id] = exchange_class({
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {
                    "defaultType": "spot",  # Avoid derivatives endpoints
                },
            })
        return self._active_exchanges[exchange_id]

    def fetch_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: int = None,
        retries: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from a cryptocurrency exchange.
        """
        exchange_instance = self.get_exchange(exchange)

        for attempt in range(retries):
            try:
                # Pass 'since' to the exchange to allow pagination
                ohlcv = exchange_instance.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                    since=since
                )

                if not ohlcv:
                    return pd.DataFrame()

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

                return df

            except Exception as e:
                print(f"[WARNING] CCXT Attempt {attempt + 1} failed for {symbol}: {e}")
                time.sleep(exchange_instance.rateLimit / 1000.0 if exchange_instance.rateLimit else 3)

        raise RuntimeError(f"Failed to fetch data from {exchange} for {symbol} after {retries} retries.")