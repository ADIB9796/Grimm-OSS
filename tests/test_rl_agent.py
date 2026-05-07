import os
import numpy as np
import pandas as pd
from src.data.data_manager import DataManager
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent
from src.ai.optuna_tuner import run_optimization

# ==========================================
# CONFIGURATION TOGGLES
# ==========================================
USE_OPTUNA = False  # Set to True to search for new params, False to use Best DNA
CHECKPOINT_PATH = "models/rl_checkpoint.pth"
FINAL_MODEL_PATH = "models/rl_trading_model.pth"

# Trial 22 "Sharpe 6.00" Winner
BEST_DNA = {
    'lr': 1.700933528056293e-05, 
    'gamma': 0.9787113384975228, 
    'batch_size': 64, 
    'epsilon_decay': 0.9913064056788868
}

def main():
    print("\n[1/3] Loading Market Data...")
    manager = DataManager()
    
    data = manager.get_crypto_data(
        symbol="BTC/USDT",
        exchange="kucoin",
        timeframe="1h",
        limit=4000 
    )

    if data.empty:
        print("[ERROR] Failed to fetch data. Exiting.")
        return

    if len(data) <= 1500:
        print(f"[ERROR] Fetched only {len(data)} bars. Need more than 1500 for a proper split.")
        return

    # Hold out the last 1000 bars for final Backtest engine. Train on the rest.
    train_data = data.iloc[:-1000]

    print("\n[2/3] Hyperparameter Configuration...")
    if USE_OPTUNA:
        print("      Running Optuna Optimization...")
        best_params = run_optimization(train_data) 
        print("\n" + "="*50)
        print("OPTUNA STAGE COMPLETE")
        print(f"   Best Parameters Found: {best_params}")
        print("="*50)
    else:
        print("      Skipping Optuna. Loading 'Turbo' DNA...")
        best_params = BEST_DNA

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
    start_episode = 0

    # --- CHECKPOINT LOADING LOGIC ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n[INFO] Found existing checkpoint at {CHECKPOINT_PATH}")
        start_episode = agent.load_checkpoint(CHECKPOINT_PATH)
        print(f"       Resuming from Episode {start_episode} with Epsilon: {agent.epsilon:.4f}")

    for e in range(start_episode, episodes):
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

        # --- CHECKPOINT SAVING LOGIC (Every 50 episodes) ---
        if (e + 1) % 50 == 0:
            agent.save_checkpoint(CHECKPOINT_PATH, e + 1)
            print(f"      [System] Checkpoint saved at episode {e+1}")

    # Save final production model and clean up checkpoint
    agent.save(FINAL_MODEL_PATH)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        
    print("\nTRAINING FINISHED!")
    print(f"      Production Model saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()