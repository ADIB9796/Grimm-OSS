import MetaTrader5 as mt5
import pandas as pd

class MT5Provider:
    def __init__(self):
        if not mt5.initialize():
            raise ConnectionError(
                f"Failed to initialize MetaTrader 5. Error code: {mt5.last_error()}\n"
                "Ensure MT5 is installed, running, and explicitly allowed in settings."
            )

    def fetch_data(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, bars=500) -> pd.DataFrame:
        """Fetch OHLCV data from MetaTrader 5."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

        if rates is None or len(rates) == 0:
            raise ValueError(f"No data retrieved for {symbol}. Check if symbol exists in MT5 Market Watch.")

        df = pd.DataFrame(rates)
        df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def shutdown(self):
        """Shut down MT5 connection."""
        mt5.shutdown()