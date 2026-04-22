import numpy as np

class RiskManager:
    """
    Handles risk management, position sizing, and capital protection.
    """

    def __init__(
        self,
        balance: float,
        risk_per_trade: float = 0.02,
        max_drawdown: float = 0.20,
    ):
        self.initial_balance = balance
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.peak_balance = balance

    # -------------------------------------------------------------
    # POSITION SIZING
    # -------------------------------------------------------------
    def calculate_position_size(self, entry_price, stop_loss_price, min_size=0.001):
        """
        Calculates position size based on risk per trade, ensuring 
        it does not exceed purchasing power or fall below exchange minimums.
        """
        if self.balance <= 0 or not self.can_trade():
            return 0.0

        risk_amount = self.balance * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            return 0.0

        # 1. Calculate ideal theoretical size
        position_size = risk_amount / risk_per_unit

        # 2. Purchasing Power Check (No Margin/Leverage assumed)
        max_affordable_size = self.balance / entry_price
        position_size = min(position_size, max_affordable_size)

        # 3. Minimum Size Check (Prevents exchange rejection for dust trades)
        if position_size < min_size:
            return 0.0  

        return float(position_size)

    # -------------------------------------------------------------
    # STOP-LOSS & TAKE-PROFIT
    # -------------------------------------------------------------
    def calculate_stop_loss(self, entry_price, risk_percent=0.01):
        """
        Default fixed-percentage stop loss for Long positions.
        """
        return entry_price * (1 - risk_percent)

    def calculate_atr_stop_loss(self, entry_price, current_atr, multiplier=2.0):
        """
        Sets a dynamic stop loss based on current market volatility (ATR).
        multiplier: How many 'units' of volatility you are willing to risk.
        """
        return entry_price - (current_atr * multiplier)

    def calculate_take_profit(self, entry_price, stop_loss_price, reward_ratio=2.0):
        """
        Calculates Take-Profit based on the actual distance to Stop-Loss.
        """
        risk_distance = abs(entry_price - stop_loss_price)
        return entry_price + (risk_distance * reward_ratio)

    # -------------------------------------------------------------
    # DRAWDOWN CONTROL
    # -------------------------------------------------------------
    def update_balance(self, new_balance):
        self.balance = new_balance
        self.peak_balance = max(self.peak_balance, new_balance)

    def current_drawdown(self):
        """
        Calculates current drawdown as a positive percentage (e.g., 0.15 = 15%).
        """
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance

    def can_trade(self):
        """
        Prevent trading if maximum drawdown limit is breached.
        """
        return self.current_drawdown() <= self.max_drawdown