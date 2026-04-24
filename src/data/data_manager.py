import pandas as pd
import numpy as np
from .ccxt_provider import CCXTProvider
from .yfinance_provider import YFinanceProvider

try:
    from .mt5_provider import MT5Provider
except ImportError:
    MT5Provider = None

class DataManager:
    def __init__(self):
        self.ccxt = CCXTProvider()
        self.yf = YFinanceProvider()
        self.mt5 = None

    def get_crypto_data(self, exchange, symbol, timeframe="1h", limit=500):
        df = self.ccxt.fetch_data(exchange, symbol, timeframe, limit)
        return self.add_technical_indicators(df)

    def get_stock_data(self, symbol, start, end, interval="1d"):
        df = self.yf.fetch_data(symbol, start, end, interval)
        return self.add_technical_indicators(df)

    def get_forex_data(self, symbol, timeframe, bars=500):
        if MT5Provider is None:
            raise EnvironmentError("MetaTrader5 not available.")
        if self.mt5 is None:
            self.mt5 = MT5Provider()
        df = self.mt5.fetch_data(symbol, timeframe, bars)
        return self.add_technical_indicators(df)

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pure Pandas/Numpy implementation. No external dependencies required.
        """
        df = df.copy()
        
        # 1. RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # 2. ATR (14)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATRr_14'] = tr.rolling(14).mean()

        # 3. SMA (20)
        df['SMA_20'] = df['close'].rolling(20).mean()

        # 4. Volatility (20)
        df['STDEV_20'] = df['close'].rolling(20).std()

        # 5. Price Change
        df['PCTRET_1'] = df['close'].pct_change()

        # Drop NaNs created by rolling windows
        df = df.dropna()
        
        # Enforce exact 10-column set for Transformer compatibility
        cols = ['open', 'high', 'low', 'close', 'volume', 'RSI_14', 'ATRr_14', 'SMA_20', 'STDEV_20', 'PCTRET_1']
        return df[cols]