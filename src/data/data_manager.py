from .ccxt_provider import CCXTProvider
from .yfinance_provider import YFinanceProvider

# Attempt to import MT5Provider safely
try:
    from .mt5_provider import MT5Provider
except ImportError:
    MT5Provider = None


class DataManager:
    """Centralized data manager for all market data sources."""

    def __init__(self):
        self.ccxt = CCXTProvider()
        self.yf = YFinanceProvider()
        self.mt5 = None  # Initialized only when needed

    # ------------------------------------------------------------------
    # CRYPTO DATA (CCXT)
    # ------------------------------------------------------------------
    def get_crypto_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ):
        return self.ccxt.fetch_data(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # STOCK DATA (YAHOO FINANCE)
    # ------------------------------------------------------------------
    def get_stock_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ):
        return self.yf.fetch_data(symbol, start, end, interval)

    # ------------------------------------------------------------------
    # FOREX DATA (META TRADER 5)
    # ------------------------------------------------------------------
    def get_forex_data(self, symbol, timeframe, bars=500):
        """Fetch forex data from MetaTrader 5."""
        if MT5Provider is None:
            raise EnvironmentError(
                "MetaTrader5 is not available in this environment. "
                "Run this feature on Windows with MetaTrader 5 installed."
            )

        if self.mt5 is None:
            self.mt5 = MT5Provider()

        return self.mt5.fetch_data(symbol, timeframe, bars)