import MetaTrader5 as mt5
import pandas as pd


class MT5Provider:
    def __init__(self):
        if not mt5.initialize():
            raise ConnectionError(
                "Failed to initialize MetaTrader 5. "
                "Ensure MT5 is installed and running."
            )

    def fetch_data(self, symbol="EURUSD",
                   timeframe=mt5.TIMEFRAME_H1,
                   bars=500) -> pd.DataFrame:
        """Fetch OHLCV data from MetaTrader 5."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

        if rates is None:
            raise ValueError(f"No data retrieved for {symbol}")

        df = pd.DataFrame(rates)
        df.rename(columns={"time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        return df[[
            "timestamp", "open", "high", "low",
            "close", "tick_volume"
        ]].rename(columns={"tick_volume": "volume"})

    def shutdown(self):
        """Shut down MT5 connection."""
        mt5.shutdown()