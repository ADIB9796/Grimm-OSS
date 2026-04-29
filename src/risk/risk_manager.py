import numpy as np

class RiskManager:
    """
    Handles risk management, position sizing via Kelly Criterion, 
    and capital protection adapted for HIGH LEVERAGE brokers (Exness).
    """
    def __init__(
        self,
        balance: float,
        fractional_kelly: float = 0.1,
        max_drawdown: float = 0.20,
        max_position_size_pct: float = 0.10, # Max total EXPOSURE, not margin
        leverage: float = 1.0,               # Default Exness Leverage
        is_training: bool = False            # SAFETY TOGGLE
    ):
        self.initial_balance = balance
        self.balance = balance
        self.fractional_kelly = fractional_kelly
        self.max_drawdown = max_drawdown
        
        self.max_position_size_pct = max_position_size_pct 
        self.leverage = leverage
        self.is_training = is_training
        
        # TRAINING MODE OVERRIDE: Cap leverage at 10x during learning phase
        # so the agent isn't instantly liquidated by early random actions.
        self.active_leverage = min(leverage, 10.0) if self.is_training else leverage
        
        self.peak_balance = balance
        self.win_history = [] 

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
        if not self.win_history:
            return 1.0 # Default starting ratio
            
        wins = [r for r in self.win_history if r > 0]
        losses = [abs(r) for r in self.win_history if r < 0]
        
        avg_win = np.mean(wins) if wins else 0.01
        avg_loss = np.mean(losses) if losses else 0.01
        
        return avg_win / avg_loss

    def can_trade(self):
        """Blocks trading if drawdown exceeds maximum allowed."""
        if self.peak_balance == 0: return True
        current_dd = (self.peak_balance - self.balance) / self.peak_balance
        return current_dd < self.max_drawdown

    # -------------------------------------------------------------
    # POSITION SIZING (KELLY CRITERION - LEVERAGE ADJUSTED)
    # -------------------------------------------------------------
    def get_kelly_size(self, win_probability: float):
        """
        Calculates optimal margin allocation based on Kelly Criterion,
        adjusted so that leveraged exposure doesn't blow the account.
        Returns: Percentage of BALANCE to use as MARGIN.
        """
        if not self.can_trade():
            return 0.0

        b = self.calculate_b_ratio()
        p = win_probability
        q = 1 - p
        
        # Base Kelly Formula
        kelly_f = (b * p - q) / (b + 1e-8)
        
        if kelly_f <= 0:
            return 0.0
            
        # 1. Get raw exposure size
        target_exposure_pct = kelly_f * self.fractional_kelly
        
        # 2. Cap maximum exposure
        target_exposure_pct = min(target_exposure_pct, self.max_position_size_pct)
        
        # 3. Translate Exposure into required Margin based on the ACTIVE Leverage
        margin_pct = target_exposure_pct / self.active_leverage
        
        return margin_pct

    # -------------------------------------------------------------
    # STOP-LOSS
    # -------------------------------------------------------------
    def calculate_atr_stop_loss(self, entry_price, current_atr, multiplier=2.0, side='long'):
        """Sets dynamic stop loss based on ATR."""
        if side == 'long':
            return entry_price - (current_atr * multiplier)
        return entry_price + (current_atr * multiplier)