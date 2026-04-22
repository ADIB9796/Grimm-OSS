import numpy as np
from src.backtesting.metrics import (
    calculate_total_return,
    calculate_win_rate,
    calculate_max_drawdown,
    calculate_sharpe_ratio
)

class Backtester:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        # Force the agent to only use its brain (no random guessing during test)
        self.agent.epsilon = 0.0 

    def run(self):
        state, _ = self.env.reset()
        done = False
        
        portfolio_history = [self.env.initial_balance]
        trade_returns = []
        
        print("\nRunning Backtest Engine...")
        
        while not done:
            # Agent predicts the best move
            action = self.agent.act(state)
            
            # Execute in environment
            next_state, reward, done, _, _ = self.env.step(action)
            
            # Track portfolio value
            current_balance = self.env.risk_manager.balance
            portfolio_history.append(current_balance)
            
            # If a trade just closed, record its return
            if len(self.env.returns) > len(trade_returns):
                trade_returns.append(self.env.returns[-1])
            
            state = next_state

        return self._generate_report(portfolio_history, trade_returns)

    def _generate_report(self, portfolio_history, trade_returns):
        initial = portfolio_history[0]
        final = portfolio_history[-1]
        
        metrics = {
            "Initial Balance": f"${initial:,.2f}",
            "Final Balance": f"${final:,.2f}",
            "Total Return": f"{calculate_total_return(initial, final):.2f}%",
            "Win Rate": f"{calculate_win_rate(trade_returns):.2f}%",
            "Max Drawdown": f"{calculate_max_drawdown(portfolio_history):.2f}%",
            "Sharpe Ratio": f"{calculate_sharpe_ratio(trade_returns):.2f}",
            "Total Trades": len(trade_returns)
        }
        return metrics