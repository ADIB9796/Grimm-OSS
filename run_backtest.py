import torch
from src.data.data_manager import DataManager
from src.ai.trading_env import TradingEnvironment
from src.ai.rl_agent import RLAgent
from src.backtesting.backtester import Backtester

def main():
    print("=== GRIMM-OSS BACKTESTING ENGINE ===")
    
    # 1. Load Unseen Data (The true test)
    manager = DataManager()
    data = manager.get_crypto_data(symbol="BTC/USD", exchange="kraken", timeframe="1h", limit=2500)
    
    # We used [:-500] for training, so we use [-500:] for the backtest
    test_data = data.iloc[-500:].copy() 
    
    # 2. Setup Environment
    env = TradingEnvironment(test_data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 3. Load the Trained Agent
    print("Loading Trained RL Model...")
    agent = RLAgent(state_size=state_size, action_size=action_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        agent.model.load_state_dict(torch.load("models/rl_trading_model.pth", map_location=device, weights_only=True))
        agent.model.eval() # Set network to evaluation mode
        print("RL Model Loaded Successfully.")
    except FileNotFoundError:
        print("ERROR: 'models/rl_trading_model.pth' not found. Did you run the training script first?")
        return

    # 4. Run Backtest
    backtester = Backtester(env, agent)
    results = backtester.run()
    
    # 5. Print Tear Sheet
    print("\n" + "="*40)
    print("BACKTEST PERFORMANCE REPORT")
    print("="*40)
    for key, value in results.items():
        print(f"{key:<20}: {value}")
    print("="*40)

if __name__ == "__main__":
    main()