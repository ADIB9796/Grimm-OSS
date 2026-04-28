import optuna
import numpy as np
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent
from src.data.data_manager import DataManager

# Force Optuna to output its default logs alongside our custom prints
optuna.logging.set_verbosity(optuna.logging.INFO)

def walk_forward_objective(trial, data):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.999)

    size = len(data)
    folds = [
        (data.iloc[0 : int(size*0.5)], data.iloc[int(size*0.5) : int(size*0.75)]),
        (data.iloc[0 : int(size*0.75)], data.iloc[int(size*0.75) : size])
    ]
    
    fold_rewards = []

    for fold_idx, (train_data, val_data) in enumerate(folds):
        train_env = TradingEnvironment(train_data)
        val_env = TradingEnvironment(val_data)
        
        state_size = train_env.observation_space.shape[0]
        action_size = train_env.action_space.n

        agent = RLAgent(state_size, action_size, lr=lr, gamma=gamma, 
                        epsilon_decay=epsilon_decay, batch_size=batch_size)

        # 1. Search Phase: Short, aggressive training bursts
        episodes = 50 
        
        for e in range(episodes):
            # 2. Deterministic Execution: Fixed seed
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

            # 3. Heartbeat Logging
            if (e + 1) % 10 == 0:
                print(f"      [Trial {trial.number}] Fold {fold_idx+1} | Episode {e+1:02d}/{episodes} | Reward: {episode_reward:6.2f}")

            trial.report(episode_reward, step=(fold_idx * episodes + e))
            if trial.should_prune():
                print(f"      [Trial {trial.number}] Pruned by MedianPruner.")
                raise optuna.TrialPruned()

        print(f"      [Trial {trial.number}] Fold {fold_idx+1}/{len(folds)} Internal Training Complete.")

        # Validation
        state, _ = val_env.reset(seed=42)
        done = False
        val_reward = 0
        agent.epsilon = 0.0 
        
        while not done:
            action = agent.act(state)
            state, reward, done, _, _ = val_env.step(action)
            val_reward += reward
        
        print(f"      [Trial {trial.number}] Fold {fold_idx+1} Validation Reward: {val_reward:6.2f}")
        fold_rewards.append(val_reward)
            
    return np.mean(fold_rewards)

def run_optimization(data, n_trials=50):
    print(f"[INFO] Initializing 2-Fold Walk-Forward Optimization")
    
    # 4. Patient Pruning: Give the agent 25 episodes to learn the basics
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=25)
    
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: walk_forward_objective(trial, data), n_trials=n_trials)
    
    return study.best_params

if __name__ == "__main__":
    dm = DataManager()
    data = dm.get_crypto_data("kucoin", "BTC/USDT", "1h", 4000)
    
    if len(data) > 1000:
        run_optimization(data)
    else:
        print("[ERROR] Insufficient data fetched for optimization.")