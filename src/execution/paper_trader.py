import pandas as pd
from typing import Dict, Any

class PaperTrader:
    """
    Simulates Exness real-world execution.
    Features: Zero commission, Spread-based costs (slippage), and Leverage.
    """
    def __init__(self, initial_balance: float = 10000.0, fee_rate: float = 0.0, slippage_pct: float = 0.0015, leverage: float = 1.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Exness Specific Defaults
        self.fee_rate = fee_rate         # 0.0 for Exness Standard/Pro
        self.slippage_pct = slippage_pct # 0.15% to simulate Exness spread (worst-case)
        self.leverage = leverage         # E.g., 50.0 for 1:50 leverage
        
        # Position state
        self.position_size_asset = 0.0   # How much BTC we hold
        self.entry_price = 0.0
        self.margin_used = 0.0           # The actual USD cash locked in the trade
        self.is_in_position = False
        
        # Metrics
        self.trade_history = []
        self.total_fees_paid = 0.0

    def execute_trade(self, action: int, current_price: float, allocated_margin: float, timestamp: Any) -> Dict[str, Any]:
        """
        Executes a trade based on the signal and allocated margin.
        action: 0 (HOLD), 1 (BUY), 2 (SELL)
        """
        trade_receipt = {
            "timestamp": timestamp,
            "action": action,
            "executed_price": current_price,
            "fees": 0.0, # Exness commissions are usually zero
            "pnl": 0.0,
            "balance": self.current_balance,
            "status": "ignored"
        }

        # BUY LOGIC
        if action == 1 and not self.is_in_position and allocated_margin > 0:
            # Apply Exness Spread (Buy at Ask price, which is higher)
            executed_price = current_price * (1 + self.slippage_pct)
            
            # Apply Leverage to get total purchasing power
            purchasing_power_usd = allocated_margin * self.leverage
            
            # Calculate any flat fees (Usually 0 for Exness crypto)
            fee_usd = purchasing_power_usd * self.fee_rate
            self.total_fees_paid += fee_usd
            
            # Actual investment size in BTC
            invested_usd = purchasing_power_usd - fee_usd
            self.position_size_asset = invested_usd / executed_price
            
            # Update state (lock margin, subtract fees)
            self.current_balance -= (allocated_margin + fee_usd)
            self.margin_used = allocated_margin
            self.entry_price = executed_price
            self.is_in_position = True
            
            trade_receipt.update({
                "executed_price": executed_price,
                "fees": fee_usd,
                "status": "FILLED_BUY",
                "balance": self.current_balance
            })

        # SELL LOGIC
        elif action == 2 and self.is_in_position:
            # Apply Exness Spread (Sell at Bid price, which is lower)
            executed_price = current_price * (1 - self.slippage_pct)
            
            # Gross return on the asset
            gross_usd = self.position_size_asset * executed_price
            
            # Calculate any flat fees
            fee_usd = gross_usd * self.fee_rate
            self.total_fees_paid += fee_usd
            
            # Net return
            net_usd = gross_usd - fee_usd
            
            # Calculate PnL relative to the LEVERAGED entry
            # PnL = Exit Value - Entry Value
            entry_value_usd = self.position_size_asset * self.entry_price
            pnl = net_usd - entry_value_usd
            
            # Update state: Return the locked margin + the PnL
            self.current_balance += (self.margin_used + pnl)
            
            self.position_size_asset = 0.0
            self.entry_price = 0.0
            self.margin_used = 0.0
            self.is_in_position = False
            
            trade_receipt.update({
                "executed_price": executed_price,
                "fees": fee_usd,
                "pnl": pnl,
                "status": "FILLED_SELL",
                "balance": self.current_balance
            })
            self.trade_history.append(trade_receipt)

        return trade_receipt

    def get_equity(self, current_price: float) -> float:
        """Returns total portfolio value (cash + floating PnL)."""
        if not self.is_in_position:
            return self.current_balance
            
        # Float PnL is based on current Bid price (if we were to sell right now)
        floating_exit_price = current_price * (1 - self.slippage_pct)
        current_value = self.position_size_asset * floating_exit_price
        entry_value = self.position_size_asset * self.entry_price
        floating_pnl = current_value - entry_value
        
        # Equity = Free Balance + Locked Margin + Floating PnL
        return self.current_balance + self.margin_used + floating_pnl