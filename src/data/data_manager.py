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
        df = df.copy()
        
        # 1-5. Original Features
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATRr_14'] = tr.rolling(14).mean()

        df['SMA_20'] = df['close'].rolling(20).mean()
        df['STDEV_20'] = df['close'].rolling(20).std()
        df['PCTRET_1'] = df['close'].pct_change()

        # 6. FEATURE EXPANSION: Rolling VWAP (24 periods)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vol_price = typical_price * df['volume']
        df['VWAP_24'] = vol_price.rolling(24).sum() / (df['volume'].rolling(24).sum() + 1e-8)

        # 7. FEATURE EXPANSION: Order Book Imbalance Proxy (Money Flow / Buying Pressure)
        # Calculates how much volume is driving the close towards the high vs the low
        buy_pressure = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['VOL_IMB'] = buy_pressure * df['volume']

        df = df.dropna()
        
        # Return 12 columns (Transformer will only use the first 10, RL agent gets all 12 + internal state)
        cols = ['open', 'high', 'low', 'close', 'volume', 'RSI_14', 'ATRr_14', 'SMA_20', 'STDEV_20', 'PCTRET_1', 'VWAP_24', 'VOL_IMB']
        return df[cols]