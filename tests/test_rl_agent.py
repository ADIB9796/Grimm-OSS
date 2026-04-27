import numpy as np
import pandas as pd
from src.data.data_manager import DataManager
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent
from src.ai.optuna_tuner import run_optimization

def main():
    print("\n[1/3] Loading Market Data...")
    manager = DataManager()
    
    # ENHANCEMENT: Fetch 4000 bars for deeper learning and validation
    data = manager.get_crypto_data(
        symbol="BTC/USD",
        exchange="kraken",
        timeframe="1h",
        limit=4000 
    )

    if data.empty:
        print("[ERROR] Failed to fetch data. Exiting.")
        return

    # Hold out the last 1000 bars for final Backtest engine. Train on first 3000.
    train_data = data.iloc[:-1000]

    print("\n[2/3] Running Optuna Optimization...")
    print("      (Evaluating configurations across Walk-Forward folds)")
    
    best_params = run_optimization(train_data) 

    print("\n" + "="*50)
    print("OPTUNA STAGE COMPLETE")
    print(f"   Best Parameters Found: {best_params}")
    print("="*50)

    print("\n[3/3] STARTING FINAL TRAINING...")
    
    env = TradingEnvironment(train_data)
    state_size = env.observation_space.shape[0] 
    action_size = env.action_space.n 

    agent = RLAgent(
        state_size=state_size, 
        action_size=action_size,
        **best_params 
    )

    episodes = 2000 

    for e in range(episodes):
        state, _ = env.reset() 
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state) 
            next_state, reward, done, truncated, info = env.step(action) 
            
            agent.remember(state, action, reward, next_state, done) 
            agent.replay() 
            
            state = next_state
            total_reward += reward
        
        agent.update_epsilon() 

        if (e + 1) % 10 == 0: 
            print(f"Episode {e+1:04d}/{episodes} | Reward: {total_reward:6.2f} | Epsilon: {agent.epsilon:.4f}") 

    agent.save("models/rl_trading_model.pth")
    print("\nTRAINING FINISHED!")
    print("      Model saved to models/rl_trading_model.pth")

if __name__ == "__main__":
    main()