"""
Robust Market Data Provider
Supports Yahoo Finance, Stooq, and Local CSV fallback.
"""

import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from pathlib import Path

class YFinanceProvider:
    def fetch_data(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch historical data with multiple fallbacks."""
        yahoo_error = None
        stooq_error = None

        # 1. Yahoo Finance
        try:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                threads=False,
                timeout=10
            )
            if not data.empty:
                return self._format_data(data)
            raise ValueError("Yahoo Finance returned empty data.")
        except Exception as e:
            yahoo_error = e
            print(f"[WARNING] Yahoo Finance failed for {symbol}: {e}")

        # 2. Stooq Fallback
        try:
            print("[INFO] Falling back to Stooq...")
            stooq_symbol = f"{symbol.lower()}.us"
            url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            data = pd.read_csv(StringIO(response.text))
            if not data.empty and "Date" in data.columns:
                return self._format_data(data)
            raise ValueError("Invalid data received from Stooq.")
        except Exception as e:
            stooq_error = e
            print(f"[WARNING] Stooq failed for {symbol}: {e}")

        # 3. Local CSV Fallback
        try:
            print("[INFO] Falling back to local dataset...")
            file_path = Path("data/sample_market_data.csv")

            if not file_path.exists():
                raise FileNotFoundError(f"Local dataset not found at {file_path}")

            data = pd.read_csv(file_path)
            return self._format_data(data)

        except Exception as local_error:
            raise RuntimeError(
                f"All data sources failed for {symbol}.\n"
                f"Yahoo Error: {yahoo_error}\n"
                f"Stooq Error: {stooq_error}\n"
                f"Local Error: {local_error}"
            )

    @staticmethod
    def _format_data(data: pd.DataFrame) -> pd.DataFrame:
        """Standardize dataframe format."""
        data = data.copy()
        
        # yfinance often returns multi-index columns in recent versions, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.reset_index(inplace=True)
        data.columns = [str(col).lower() for col in data.columns]

        if "date" in data.columns:
            data.rename(columns={"date": "timestamp"}, inplace=True)
        elif "datetime" in data.columns:
            data.rename(columns={"datetime": "timestamp"}, inplace=True)

        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        data = data.loc[:, required_columns].copy()
        data.loc[:, "timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data.sort_values("timestamp", inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data