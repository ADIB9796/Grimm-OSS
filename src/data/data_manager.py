import pandas as pd
import numpy as np
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
        # 1. Get cached exchange to access timeframe math
        ex_instance = self.ccxt.get_exchange(exchange)
        
        # 2. Calculate exact milliseconds per candle based on timeframe
        tf_ms = ex_instance.parse_timeframe(timeframe) * 1000
        
        # 3. Calculate starting timestamp
        since = ex_instance.milliseconds() - (limit * tf_ms)
        
        all_df = []
        fetched = 0
        
        print(f"[INFO] Paginated fetch started for {symbol} ({limit} bars)...")
        while fetched < limit:
            chunk_limit = min(720, limit - fetched)
            df_chunk = self.ccxt.fetch_data(exchange, symbol, timeframe, chunk_limit, since=since)
            
            if df_chunk is None or df_chunk.empty:
                break
                
            all_df.append(df_chunk)
            fetched += len(df_chunk)
            
            # 4. Advance the 'since' tracker to the last fetched candle + 1ms
            last_timestamp = int(df_chunk['timestamp'].iloc[-1].timestamp() * 1000)
            since = last_timestamp + 1
            
            # Respect exchange rate limits during pagination
            rate_limit = ex_instance.rateLimit / 1000.0 if ex_instance.rateLimit else 0.1
            time.sleep(rate_limit)

        if not all_df:
            return pd.DataFrame()

        full_df = pd.concat(all_df).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return self.add_technical_indicators(full_df.tail(limit))

    def get_stock_data(self, symbol, start, end, interval="1d"):
        df = self.yf.fetch_data(symbol, start, end, interval)
        return self.add_technical_indicators(df)

    def get_forex_data(self, symbol, timeframe, bars=500):
        if MT5Provider is None: raise EnvironmentError("MetaTrader5 not available.")
        if self.mt5 is None: self.mt5 = MT5Provider()
        df = self.mt5.fetch_data(symbol, timeframe, bars)
        return self.add_technical_indicators(df)

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pure Pandas implementation of technical indicators.
        Returns 12 columns: 10 for PriceTransformer, 12 for RL Agent.
        """
        if df.empty: return df
        df = df.copy()
        
        # 1. RSI (Momentum)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # 2. ATR (Volatility)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATRr_14'] = tr.rolling(14).mean()

        # 3. VWAP (Volume-Weighted Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP_24'] = (typical_price * df['volume']).rolling(24).sum() / (df['volume'].rolling(24).sum() + 1e-8)

        # 4. VOL_IMB (Volume Imbalance)
        range_size = (df['high'] - df['low']) + 1e-8
        df['VOL_IMB'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / range_size * df['volume']

        # 5. PCTRET (Percentage Return)
        df['PCTRET_1'] = df['close'].pct_change()

        # 6-7. Trend & Dispersion
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['STDEV_20'] = df['close'].rolling(20).std()

        df = df.dropna()
        
        # Final Column Mapping
        cols = [
            'open', 'high', 'low', 'close', 'volume', 
            'RSI_14', 'ATRr_14', 'VWAP_24', 'VOL_IMB', 'PCTRET_1', 
            'SMA_20', 'STDEV_20'
        ]
        return df[cols]