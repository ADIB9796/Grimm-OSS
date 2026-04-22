import numpy as np
import pandas as pd

def calculate_total_return(initial_balance, final_balance):
    return ((final_balance - initial_balance) / initial_balance) * 100

def calculate_win_rate(trades):
    if not trades:
        return 0.0
    winning_trades = [t for t in trades if t > 0]
    return (len(winning_trades) / len(trades)) * 100

def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_dd = 0.0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
            
    return max_dd * 100

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) < 2:
        return 0.0
    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array) + 1e-8
    
    # Assuming hourly returns, annualized (24 hours * 365 days = 8760)
    annualized_factor = np.sqrt(8760) 
    return (mean_return - risk_free_rate) / std_return * annualized_factor