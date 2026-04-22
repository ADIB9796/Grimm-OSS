import pandas as pd
import ta
from .base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy."""

    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["rsi"] = ta.momentum.RSIIndicator(
            close=df["close"],
            window=self.period
        ).rsi()

        df["signal"] = 0
        df.loc[df["rsi"] < self.oversold, "signal"] = 1
        df.loc[df["rsi"] > self.overbought, "signal"] = -1

        return df