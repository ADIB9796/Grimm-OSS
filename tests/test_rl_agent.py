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
USE_OPTUNA = False  
CHECKPOINT_PATH = "models/BTC_rl_checkpoint.pth"
FINAL_MODEL_PATH = "models/BTC_rl_trading_model.pth"
FINAL_ONNX_PATH = "models/BTC_rl_trading_model.onnx"

# Top performing structural hyperparameters
BEST_DNA = {
    'lr': 1.700933528056293e-05, 
    'gamma': 0.9787113384975228, 
    'batch_size': 64, 
    'epsilon_decay': 0.9913064056788868
}

def main():
    print("\n[1/4] Extracting Core Historical Series (L2 Data Synthesizer)...")
    manager = DataManager()
    
    df_1h = manager.get_crypto_data("kucoin", "BTC/USDT", "1h", 6000)
    df_4h = manager.get_crypto_data("kucoin", "BTC/USDT", "4h", 2000)

    if df_1h.empty or df_4h.empty:
        print("[ERROR] High-density timeframe frames failed to download.")
        return

    print("\n[2/4] Merging Multi-Timeframe Matrices...")
    df_4h = df_4h.add_suffix('_4h').rename(columns={'timestamp_4h': 'timestamp'})
    
    df_merged = pd.merge_asof(
        df_1h.sort_values('timestamp'),
        df_4h.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    ).dropna()
    
    if len(df_merged) <= 1500:
        print("[ERROR] Data matrix sync range insufficient.")
        return

    df_merged = df_merged.drop(columns=['timestamp'])
    train_data = df_merged.iloc[:-1000]
    
    print(f"      [INFO] Environment Dataset prepared with {train_data.shape[1]} features.")

    print("\n[3/4] Initializing Agent Parameter Mappings...")
    if USE_OPTUNA:
        print("      Running Optuna Optimization...")
        best_params = run_optimization(train_data) 
        print("\n" + "="*50)
        print(f"   Best Parameters Found: {best_params}")
        print("="*50)
    else:
        print("      Skipping Optuna. Loading 'Turbo' DNA...")
        best_params = BEST_DNA

    print("\n[4/4] LOADING RUNTIME ENGINE ($100 ISOLATION MODE)...")
    # Hardcode initialization parameters to force $100 baseline boundaries
    env = TradingEnvironment(train_data, initial_balance=100.0, transaction_cost=0.0015, is_training=True)
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
        print(f"\n[INFO] Found active safe-state checkpoint at {CHECKPOINT_PATH}")
        start_episode = agent.load_checkpoint(CHECKPOINT_PATH)
        print(f"       Resuming training frame at Episode {start_episode}")

    for e in range(start_episode, episodes):
        state, _ = env.reset() 
        total_reward = 0
        done = False
        
        # Track Q-values for this episode
        q_value_tracker = []

        while not done:
            action = agent.act(state) 
            
            # Log the max Q-value from the agent's decision step
            q_value_tracker.append(agent.last_q_val)
            
            next_state, reward, done, truncated, info = env.step(action) 
            
            agent.remember(state, action, reward, next_state, done) 
            agent.replay() 
            
            state = next_state
            total_reward += reward
        
        agent.update_epsilon() 
        avg_q = np.mean(q_value_tracker) if q_value_tracker else 0.0

        if (e + 1) % 10 == 0: 
            print(f"Ep {e+1:04d}/{episodes} | Total Reward: {total_reward:7.2f} | Avg Q-Val: {avg_q:6.2f} | Epsilon: {agent.epsilon:.4f} | Terminal Wallet: ${env.risk_manager.balance:.2f}") 

        if (e + 1) % 50 == 0:
            agent.save_checkpoint(CHECKPOINT_PATH, e + 1)
            print(f"      [System] Checkpoint saved at episode {e+1}")

    # Final Saves & ONNX Export
    agent.save(FINAL_MODEL_PATH)
    agent.export_onnx(FINAL_ONNX_PATH)
    
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        
    print("\n[SUCCESS] Pipeline Completed.")
    print(f"      PyTorch Checkpoint saved to {FINAL_MODEL_PATH}")
    print(f"      Production ONNX Model saved to {FINAL_ONNX_PATH}")

if __name__ == "__main__":
    main()