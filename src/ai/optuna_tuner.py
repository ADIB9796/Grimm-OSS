import optuna
import numpy as np
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent
from src.data.data_manager import DataManager

def walk_forward_objective(trial, data):
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.999)

    # --- 2-FOLD ANCHORED WALK-FORWARD SETUP ---
    # Fold 1: Train on 0-50%, Validate on 50-75%
    # Fold 2: Train on 0-75%, Validate on 75-100%
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

        episodes = 100 
        for e in range(episodes):
            state, _ = train_env.reset()
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

            # PRUNING: Using Optuna's built-in logic + manual safety at ep 25
            if e == 25 and episode_reward < -500:
                raise optuna.TrialPruned()
            
            trial.report(episode_reward, step=(fold_idx * episodes + e))
            if trial.should_prune():
                raise optuna.TrialPruned()

        # --- VALIDATION (Out-of-Sample) ---
        state, _ = val_env.reset()
        done = False
        val_reward = 0
        agent.epsilon = 0.0 
        while not done:
            action = agent.act(state)
            state, reward, done, _, _ = val_env.step(action)
            val_reward += reward
        
        fold_rewards.append(val_reward)
            
    return np.mean(fold_rewards) # Average reward across both folds

def run_optimization(data, n_trials=50):
    print(f"[INFO] Initializing 2-Fold Walk-Forward Optimization on T4 GPU...")
    # MedianPruner helps kill bad hyperparameter sets early
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=15)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: walk_forward_objective(trial, data), n_trials=n_trials)
    
    print("\nBEST PARAMS FOUND:")
    print(study.best_params)
    return study.best_params

if __name__ == "__main__":
    dm = DataManager()
    data = dm.get_crypto_data("kraken", "BTC/USD", "1h", 3000)
    run_optimization(data)