import numpy as np

class RiskManager:
    """
    Handles risk management, capital protection, and contract conversion
    optimized for an extreme $100 capital boundary on Exness.
    """
    def __init__(
        self,
        balance: float = 100.0,
        fractional_kelly: float = 0.05,       # Defensive fractional Kelly
        max_drawdown: float = 0.20,           # 20% max total buffer protection ($20)
        max_position_size_pct: float = 0.02,  # Max target risk exposure fraction
        leverage: float = 400.0,              # Exness default crypto leverage (1:400)
        is_training: bool = False
    ):
        self.initial_balance = balance
        self.balance = balance
        self.fractional_kelly = fractional_kelly
        self.max_drawdown = max_drawdown
        self.max_position_size_pct = max_position_size_pct 
        self.leverage = leverage
        self.is_training = is_training
        
        # Training override: Lock leverage lower during initial random discovery 
        # phase so random actions don't generate execution anomalies.
        self.active_leverage = min(leverage, 100.0) if self.is_training else leverage
        
        self.peak_balance = balance
        self.win_history = [] 

    def update_performance(self, profit_pct: float, new_balance: float):
        self.win_history.append(profit_pct)
        if len(self.win_history) > 100:
            self.win_history.pop(0)
        self.balance = new_balance
        self.peak_balance = max(self.peak_balance, new_balance)

    def calculate_b_ratio(self):
        if not self.win_history:
            return 1.0
        wins = [r for r in self.win_history if r > 0]
        losses = [abs(r) for r in self.win_history if r < 0]
        avg_win = np.mean(wins) if wins else 0.01
        avg_loss = np.mean(losses) if losses else 0.01
        return avg_win / avg_loss

    def can_trade(self):
        if self.peak_balance == 0: return True
        current_dd = (self.peak_balance - self.balance) / self.peak_balance
        return current_dd < self.max_drawdown

    def get_kelly_size(self, win_probability: float):
        """Calculates abstract optimal allocation fraction."""
        if not self.can_trade() or win_probability < 0.65: # Sniper floor requirement
            return 0.0

        b = self.calculate_b_ratio()
        p = win_probability
        q = 1 - p
        
        kelly_f = (b * p - q) / (b + 1e-8)
        if kelly_f <= 0:
            return 0.0
            
        target_exposure_pct = kelly_f * self.fractional_kelly
        target_exposure_pct = min(target_exposure_pct, self.max_position_size_pct)
        
        return target_exposure_pct / self.active_leverage

    def calculate_lot_size(self, margin_pct: float, current_price: float, contract_size: float = 1.0):
        """
        Translates risk margin allocation into specific Exness API volumes.
        Forces the execution engine to stick strictly to the 0.01 micro-lot floor.
        """
        if margin_pct <= 0:
            return 0.0
            
        usable_margin = self.balance * margin_pct
        notional_exposure = usable_margin * self.active_leverage
        lots = notional_exposure / (contract_size * current_price)
        
        # Hard constraint enforcement: If agent chooses to trade, we lock it 
        # to the absolute minimum physical order block size accepted by Exness.
        if lots > 0:
            return 0.01
        return 0.0

    def calculate_atr_stop_loss(self, entry_price, current_atr, multiplier=2.0, side='long'):
        if side == 'long':
            return entry_price - (current_atr * multiplier)
        return entry_price + (current_atr * multiplier)