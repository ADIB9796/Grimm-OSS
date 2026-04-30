import optuna
import numpy as np
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent
from src.data.data_manager import DataManager

optuna.logging.set_verbosity(optuna.logging.INFO)

def walk_forward_objective(trial, data):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [64, 128]) # Removed 32 for speed
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.999)

    size = len(data)
    # Single Fold for Turbo Mode (70% Train, 30% Val)
    train_data = data.iloc[0 : int(size*0.70)]
    val_data = data.iloc[int(size*0.70) : size]
    
    train_env = TradingEnvironment(train_data, is_training=True)
    val_env = TradingEnvironment(val_data, is_training=True)
    
    state_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n

    agent = RLAgent(state_size, action_size, lr=lr, gamma=gamma, 
                    epsilon_decay=epsilon_decay, batch_size=batch_size)

    episodes = 25 # Reduced from 50 to 25 to halve runtime
    
    for e in range(episodes):
        state, _ = train_env.reset(seed=42)
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            episode_reward += reward
        
        agent.update_epsilon()

        if (e + 1) % 5 == 0:
            print(f"      [Trial {trial.number}] Episode {e+1:02d}/{episodes} | Scaled Reward: {episode_reward:6.2f}")

        trial.report(episode_reward, step=e)

        if trial.should_prune():
            print(f"      [Trial {trial.number}] Pruned by MedianPruner at Episode {e+1}.")
            raise optuna.TrialPruned()

    print(f"      [Trial {trial.number}] Internal Training Complete.")

    # Validation Phase
    state, _ = val_env.reset(seed=42)
    done = False
    val_reward = 0
    agent.epsilon = 0.0 
    
    while not done:
        action = agent.act(state)
        state, reward, done, _, _ = val_env.step(action)
        val_reward += reward
    
    val_returns = val_env.returns
    if len([r for r in val_returns if r != 0]) > 2:
        mean_ret = np.mean(val_returns)
        std_ret = np.std(val_returns) + 1e-8
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(24 * 365)
    else:
        sharpe_ratio = -1.0 
        
    print(f"      [Trial {trial.number}] Val Sharpe: {sharpe_ratio:6.2f} (Raw Reward: {val_reward:6.2f})")
        
    return sharpe_ratio

def run_optimization(data, n_trials=30): # Reduced from 50 to 30
    print(f"[TURBO] Initializing Single-Fold Walk-Forward Optimization...")
    # Rely entirely on Optuna's built-in intelligent MedianPruner
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: walk_forward_objective(trial, data), n_trials=n_trials)
    
    print(f"Best Sharpe: {study.best_value:.4f}")
    return study.best_params

if __name__ == "__main__":
    dm = DataManager()
    data = dm.get_crypto_data("kucoin", "BTC/USDT", "1h", 4000)
    
    if len(data) > 1000:
        run_optimization(data)
    else:
        print("[ERROR] Insufficient data fetched for optimization.")