import numpy as np

class RiskManager:
    """
    Handles risk management, position sizing via Kelly Criterion, 
    and capital protection across multi-asset classes.
    """
    def __init__(
        self,
        balance: float,
        fractional_kelly: float = 0.1,
        max_drawdown: float = 0.20,
        max_position_size: float = 0.05
    ):
        self.initial_balance = balance
        self.balance = balance
        self.fractional_kelly = fractional_kelly
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.peak_balance = balance
        self.win_history = [] # Stores percentage returns of closed trades

    # -------------------------------------------------------------
    # PERFORMANCE TRACKING
    # -------------------------------------------------------------
    def update_performance(self, profit_pct: float, new_balance: float):
        """Updates internal stats for Kelly calculation and drawdown control."""
        self.win_history.append(profit_pct)
        if len(self.win_history) > 100:
            self.win_history.pop(0)
        
        self.balance = new_balance
        self.peak_balance = max(self.peak_balance, new_balance)

    def calculate_b_ratio(self):
        """Calculates the Win/Loss ratio (average win / average loss)."""
        wins = [x for x in self.win_history if x > 0]
        losses = [abs(x) for x in self.win_history if x < 0]
        
        if not losses: return 2.0 # Default conservative ratio
        avg_win = np.mean(wins) if wins else 0.001
        avg_loss = np.mean(losses) if losses else 0.001
        return avg_win / avg_loss

    # -------------------------------------------------------------
    # POSITION SIZING (KELLY CRITERION)
    # -------------------------------------------------------------
    def get_kelly_size(self, win_probability: float):
        """
        Calculates optimal size using: f* = p - (1 - p) / b
        p = win probability (from Transformer), b = Win/Loss ratio.
        """
        if not self.can_trade():
            return 0.0

        b = self.calculate_b_ratio()
        p = win_probability
        q = 1 - p
        
        # Kelly Formula
        kelly_f = (b * p - q) / (b + 1e-8)
        
        if kelly_f <= 0:
            return 0.0
            
        # Apply Fractional Kelly and hard cap
        size = kelly_f * self.fractional_kelly
        return min(size, self.max_position_size)

    # -------------------------------------------------------------
    # STOP-LOSS & DRAWDOWN
    # -------------------------------------------------------------
    def calculate_atr_stop_loss(self, entry_price, current_atr, multiplier=2.0, side='long'):
        """Sets dynamic stop loss based on ATR."""
        if side == 'long':
            return entry_price - (current_atr * multiplier)
        return entry_price + (current_atr * multiplier)

    def current_drawdown(self):
        if self.peak_balance == 0: return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance

    def can_trade(self):
        return self.current_drawdown() <= self.max_drawdown