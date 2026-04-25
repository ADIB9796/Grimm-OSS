import numpy as np
import pandas as pd
from src.data.data_manager import DataManager
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent
from src.ai.optuna_tuner import run_optimization

def main():
    # 1. SETUP DATA
    print("\n[1/3] Loading Market Data...")
    manager = DataManager()
    
    # Using a 2,500 limit to provide enough room for the walk-forward splits
    data = manager.get_crypto_data(
        symbol="BTC/USD",
        exchange="kraken",
        timeframe="1h",
        limit=2500 
    )

    # Split data to keep the most recent 500 candles as a completely untouched test set
    train_data = data.iloc[:-500]

    # 2. HYPERPARAMETER TUNING STAGE
    print("\n[2/3] Running Optuna Optimization...")
    # Calculation: 50 trials * 2 folds * 100 episodes = 10,000 total episodes
    print("      (This will evaluate 50 different configurations across 10,000 total episodes)")
    
    best_params = run_optimization(train_data) # cite: [uploaded:optuna_tunar.py]

    print("\n" + "="*50)
    print("OPTUNA STAGE COMPLETE")
    print(f"   Best Parameters Found: {best_params}")
    print("="*50)

    # 3. FINAL TRAINING STAGE
    print("\n[3/3] STARTING FINAL TRAINING...")
    print("      Status: The agent is now learning using optimized settings.")

    # Initialize environment and agent using the best parameters discovered
    env = TradingEnvironment(train_data)
    state_size = env.observation_space.shape[0] # cite: [uploaded:test_rl_agent.py]
    action_size = env.action_space.n # cite: [uploaded:test_rl_agent.py]

    # Use ** unpacking to pass lr, gamma, batch_size, and epsilon_decay directly
    agent = RLAgent(
        state_size=state_size, 
        action_size=action_size,
        **best_params 
    )

    episodes = 2000 # cite: [uploaded:test_rl_agent.py]

    for e in range(episodes):
        state, _ = env.reset() # cite: [uploaded:test_rl_agent.py]
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state) # cite: [uploaded:test_rl_agent.py]
            
            # Unpacking the 5-value return from newer Gymnasium environments
            next_state, reward, done, truncated, info = env.step(action) # cite: [uploaded:test_rl_agent.py]
            
            agent.remember(state, action, reward, next_state, done) # cite: [uploaded:test_rl_agent.py]
            agent.replay() # cite: [uploaded:test_rl_agent.py]
            
            state = next_state
            total_reward += reward
        
        # Apply the optimized epsilon decay found by Optuna
        agent.update_epsilon() # cite: [uploaded:test_rl_agent.py]

        # Terminal status update every 10 episodes
        if (e + 1) % 10 == 0: # cite: [uploaded:test_rl_agent.py]
            print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}") # cite: [uploaded:test_rl_agent.py]

    # Save the final optimized model
    agent.save("models/rl_trading_model.pth")
    print("\nTRAINING FINISHED!")
    print("      Model saved to models/rl_trading_model.pth")

if __name__ == "__main__":
    main()