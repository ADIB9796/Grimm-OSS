import pandas as pd
import numpy as np
import pandas_ta as ta
import time
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

    def get_crypto_data(self, exchange, symbol, timeframe="1h", limit=3000):
        """
        Fetches crypto data with a recursive loop to bypass the standard 720-bar 
        limit of most exchanges (like Kraken).
        """
        all_df = []
        fetched = 0
        # Calculate start point: limit * timeframe in ms (approx for 1h)
        since = self.ccxt.exchange.milliseconds() - (limit * 60 * 60 * 1000) 
        
        print(f"[INFO] Paginated fetch started for {symbol} ({limit} bars)...")
        
        while fetched < limit:
            # Fetch in chunks of 720 (standard limit)
            chunk_limit = min(720, limit - fetched)
            df_chunk = self.ccxt.fetch_data(exchange, symbol, timeframe, chunk_limit)
            
            if df_chunk.empty:
                break
                
            all_df.append(df_chunk)
            fetched += len(df_chunk)
            
            # Update 'since' to the last timestamp to get the next window
            # Assumes index is datetime or 'timestamp' column exists
            time.sleep(0.1) # Rate limit protection

        full_df = pd.concat(all_df).drop_duplicates().sort_index()
        return self.add_technical_indicators(full_df.tail(limit))

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
        Vectorized technical analysis. 
        Returns 12 columns: First 10 for PriceTransformer, all 12 for RL Agent.
        """
        df = df.copy()
        
        # 1. RSI (Momentum)
        df['RSI_14'] = ta.rsi(df['close'], length=14)

        # 2. ATR (Volatility)
        df['ATRr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # 3. VWAP (Volume-Weighted Price)
        # Using 24-period rolling for intraday context
        df['VWAP_24'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'], anchor="D") 
        # Fallback if anchor fails on specific timeframes
        if df['VWAP_24'].isnull().all():
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['VWAP_24'] = (typical_price * df['volume']).rolling(24).sum() / (df['volume'].rolling(24).sum() + 1e-8)

        # 4. VOL_IMB (Volume Imbalance / Buy Pressure)
        # Ratio of where the candle closed relative to its range, weighted by volume
        range_size = (df['high'] - df['low']) + 1e-8
        df['VOL_IMB'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / range_size * df['volume']

        # 5. PCTRET (Percentage Return)
        df['PCTRET_1'] = df['close'].pct_change()

        # 6-7. Trend & Dispersion (For RL Agent internal state)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['STDEV_20'] = df['close'].rolling(20).std()

        # Clean NaNs
        df = df.dropna()
        
        # --- FINAL COLUMN MAPPING ---
        # 1-10: Used by Transformer | 1-12: Used by RL Agent
        cols = [
            'open', 'high', 'low', 'close', 'volume',  # OHLCV (5)
            'RSI_14', 'ATRr_14', 'VWAP_24', 'VOL_IMB', 'PCTRET_1', # Alpha Features (5)
            'SMA_20', 'STDEV_20' # State Context (2)
        ]
        
        return df[cols]