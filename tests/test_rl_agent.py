import os
import numpy as np
import pandas as pd
from src.data.data_manager import DataManager
from src.env.trading_env import TradingEnvironment
from src.agents.rl_agent import RLAgent
from src.ai.optuna_tuner import run_optimization

# ==========================================
# CONFIGURATION TOGGLES
# ==========================================
USE_OPTUNA = False  
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
    print("\n[1/4] Fetching Multi-Timeframe Data (L2 Synthesis)...")
    manager = DataManager()
    
    df_1h = manager.get_crypto_data("kucoin", "BTC/USDT", "1h", 6000)
    df_4h = manager.get_crypto_data("kucoin", "BTC/USDT", "4h", 2000)

    if df_1h.empty or df_4h.empty:
        print("[ERROR] Failed to fetch data. Exiting.")
        return

    print("\n[2/4] Merging Timeframes & Aligning Order Book States...")
    df_4h = df_4h.add_suffix('_4h').rename(columns={'timestamp_4h': 'timestamp'})
    
    df_merged = pd.merge_asof(
        df_1h.sort_values('timestamp'),
        df_4h.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    ).dropna()
    
    if len(df_merged) <= 1500:
        print(f"[ERROR] Synced only {len(df_merged)} bars. Need >1500 for a proper split.")
        return

    # Drop timestamp to pass a pure 56-feature float array to the env
    df_merged = df_merged.drop(columns=['timestamp'])
    train_data = df_merged.iloc[:-1000]
    
    print(f"      [INFO] Environment Dataset prepared with {train_data.shape[1]} features.")

    print("\n[3/4] Hyperparameter Configuration...")
    if USE_OPTUNA:
        print("      Running Optuna Optimization...")
        best_params = run_optimization(train_data) 
        print("\n" + "="*50)
        print(f"   Best Parameters Found: {best_params}")
        print("="*50)
    else:
        print("      Skipping Optuna. Loading 'Turbo' DNA...")
        best_params = BEST_DNA

    print("\n[4/4] STARTING L2 REINFORCEMENT TRAINING...")
    
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

        if (e + 1) % 50 == 0:
            agent.save_checkpoint(CHECKPOINT_PATH, e + 1)
            print(f"      [System] Checkpoint saved at episode {e+1}")

    agent.save(FINAL_MODEL_PATH)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        
    print("\nTRAINING FINISHED!")
    print(f"      Production Model saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()