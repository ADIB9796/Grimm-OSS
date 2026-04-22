import optuna
import numpy as np
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent

def walk_forward_split(data, n_splits=3):
    split_size = len(data) // (n_splits + 1)
    splits = []

    for i in range(n_splits):
        train_end = split_size * (i + 1)
        test_end = split_size * (i + 2)

        train = data[:train_end]
        test = data[train_end:test_end]

        splits.append((train, test))

    return splits

def evaluate_agent(agent, env, steps=500):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    # Force exploitation during evaluation
    agent.epsilon = 0.0

    step_count = 0
    while not done and step_count < steps:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        step_count += 1

    return total_reward

def objective(trial, data):
    # Hyperparameters to tune
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.990, 0.999)

    splits = walk_forward_split(data, n_splits=3)
    scores = []

    for train_data, test_data in splits:
        train_env = TradingEnvironment(train_data)
        test_env = TradingEnvironment(test_data)

        # Initialize the Agent with trial parameters
        agent = RLAgent(
            state_size=train_env.observation_space.shape[0],
            action_size=3,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            epsilon_decay=epsilon_decay
        )

        # Short training loop for optimization trials
        episodes = 30 
        for _ in range(episodes):
            state, _ = train_env.reset()
            done = False
            
            # Limit steps to speed up tuning
            step_count = 0
            while not done and step_count < 500:
                action = agent.act(state)
                next_state, reward, done, _, _ = train_env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                step_count += 1
                
            agent.update_epsilon()

        # Evaluate performance on the "future" window
        score = evaluate_agent(agent, test_env)
        scores.append(score)

    return np.mean(scores)

def run_optimization(data):
    print("[INFO] Starting Optuna Hyperparameter Optimization...")
    study = optuna.create_study(direction="maximize")
    
    # Run 20 trials to find the best configuration
    study.optimize(lambda trial: objective(trial, data), n_trials=20)

    print("\n🔥 BEST PARAMS:")
    print(study.best_params)

    return study.best_params

if __name__ == "__main__":
    from src.data.data_manager import DataManager
    
    # Fetch data to run the tuner directly
    manager = DataManager()
    data = manager.get_crypto_data(
        symbol="BTC/USD",
        exchange="kraken",
        timeframe="1h",
        limit=2000 
    )
    
    best_parameters = run_optimization(data)