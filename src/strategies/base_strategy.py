from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Returns:
            DataFrame with a 'signal' column:
            1 = Buy, -1 = Sell, 0 = Hold
        """
        pass