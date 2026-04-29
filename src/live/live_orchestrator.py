import time
import logging
import os
import numpy as np
import pandas as pd
import torch

from src.data.data_manager import DataManager
from src.ai.rl_agent import RLAgent
from src.risk.risk_manager import RiskManager
from src.execution.paper_trader import PaperTrader
from src.models.price_predictor import PriceTransformer

# Setup basic logging to the requested folder
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/trades.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LiveOrchestrator:
    """
    The Brain: Connects Data -> AI -> Risk -> Execution.
    """
    def __init__(self, rl_model_path: str = "models/rl_trading_model.pth", 
                 price_model_path: str = "models/price_model.pth", 
                 initial_capital: float = 10000.0):
        
        logging.info("Initializing Live Orchestrator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize Sub-Systems
        self.data_manager = DataManager()
        # Adjusted for Exness-style High Leverage (e.g., 50x)
        self.risk_manager = RiskManager(balance=initial_capital, leverage=50.0) 
        self.paper_trader = PaperTrader(initial_balance=initial_capital, fee_rate=0.0, slippage_pct=0.0015, leverage=50.0)
        
        # 2. Load the AI Brains
        # 10 features + 3 transformer predictions = 13
        self.state_size = 13  
        self.action_size = 3  # HOLD, BUY, SELL
        self.seq_len = 50     # Transformer sequence length
        
        # Load RL Agent
        self.agent = RLAgent(self.state_size, self.action_size)
        if os.path.exists(rl_model_path):
            self.agent.model.load_state_dict(torch.load(rl_model_path, map_location=self.device, weights_only=True))
            self.agent.model.to(self.device)
            self.agent.model.eval() # No learning in production
            self.agent.epsilon = 0.0 # No random exploration
            logging.info(f"Successfully loaded Master Model from {rl_model_path}")
        else:
            logging.warning(f"No Master Model found at {rl_model_path}. Proceeding with random weights (WARNING).")

        # Load Price Transformer
        self.transformer = PriceTransformer(input_size=10).to(self.device)
        if os.path.exists(price_model_path):
            self.transformer.load_state_dict(torch.load(price_model_path, map_location=self.device, weights_only=True))
            self.transformer.eval()
            logging.info(f"Successfully loaded Price Transformer from {price_model_path}")
        else:
            logging.warning(f"No Price Transformer found at {price_model_path}. Proceeding with random weights (WARNING).")

        self.current_pred = np.array([0.0, 1.0, 0.0]) # Default Neutral
        self.last_known_price = 0.0

    def _prepare_features(self, df):
        """Replicates the exact feature engineering from TradingEnvironment."""
        df = df.copy().select_dtypes(include=[np.number])
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["ma_fast"] = df["close"].rolling(5).mean()
        df["ma_slow"] = df["close"].rolling(20).mean()
        df["volatility"] = df["log_return"].rolling(10).std()
        df = df.dropna()

        # Normalization
        for col in df.columns:
            std = df[col].std()
            df[col] = 0 if std == 0 or np.isnan(std) else (df[col] - df[col].mean()) / (std + 1e-8)
                
        return df

    def get_current_state(self):
        """Fetches latest candles and formats them for the models."""
        # Fetch 100 bars to ensure we have enough data after dropping NaNs from rolling indicators
        df = self.data_manager.get_crypto_data("kucoin", "BTC/USDT", "1h", limit=100)
        
        self.last_known_price = df.iloc[-1]['close']
        timestamp = df.index[-1]
        
        features_df = self._prepare_features(df)
        
        if len(features_df) < self.seq_len:
            raise ValueError(f"Not enough valid data points. Expected {self.seq_len}, got {len(features_df)}")
        
        # Get the sequence of the last 50 bars for the Transformer
        sequence = features_df.iloc[-self.seq_len:].values.astype(np.float32)
        
        # 1. Get Transformer Prediction
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            self.current_pred = self.transformer(seq_tensor).cpu().numpy()[0]
            
        # 2. Build Full State for RL Agent (Current features + Predictions)
        transformer_input = sequence[-1][:10] # The most recent feature row
        full_state = np.concatenate([transformer_input, self.current_pred]).astype(np.float32)
        
        return full_state, self.last_known_price, timestamp

    def run_step(self):
        """A single iteration of the live bot loop."""
        try:
            # 1. Data Layer: Pull and Process
            state, current_price, timestamp = self.get_current_state()
            
            # 2. AI Layer: Decide Action
            action = self.agent.act(state)
            
            # 3. Risk Layer: Approve & Size based on Transformer Confidence
            win_prob = self.current_pred[2] # Index 2 is the 'UP' probability
            allowed_size_pct = self.risk_manager.get_kelly_size(win_probability=win_prob)
            
            # Translate % to actual USD Margin (Exness style)
            allocated_margin = self.risk_manager.balance * allowed_size_pct
            
            # 4. Execution Layer: Trade
            receipt = self.paper_trader.execute_trade(
                action=action,
                current_price=current_price,
                allocated_margin=allocated_margin,
                timestamp=timestamp
            )
            
            # 5. Feedback Loop: Update Risk Manager with actual PnL
            if receipt["status"] != "ignored":
                # Only log performance when a position is closed to assess the Win/Loss
                if receipt["status"] == "FILLED_SELL":
                    profit_pct = (receipt["pnl"] / allocated_margin) if allocated_margin > 0 else 0.0
                    self.risk_manager.update_performance(
                        profit_pct=profit_pct,
                        new_balance=self.paper_trader.current_balance
                    )
                
                # 6. Logging
                log_msg = f"ACTION: {receipt['status']} | PRICE: ${receipt['executed_price']:.2f} | PNL: ${receipt.get('pnl', 0.0):.2f} | EQUITY: ${self.paper_trader.get_equity(current_price):.2f}"
                print(f"[LIVE] {log_msg}")
                logging.info(log_msg)

        except Exception as e:
            logging.error(f"Critical error in execution loop: {str(e)}")

    def start_bot(self, interval_seconds=3600):
        """Runs the orchestrator in a continuous loop."""
        print("\n" + "="*50)
        print("STARTING LIVE ORCHESTRATOR")
        print("="*50)
        logging.info("Live Orchestrator started.")
        
        try:
            while True:
                self.run_step()
                time.sleep(interval_seconds) # Wait for the next candle
        except KeyboardInterrupt:
            print("\n[INFO] Bot stopped by user.")
            logging.info("Live Orchestrator stopped by user.")
            print(f"Final Equity: ${self.paper_trader.get_equity(current_price=self.last_known_price):.2f}")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Initialize and run
    orchestrator = LiveOrchestrator(
        rl_model_path="models/rl_trading_model.pth",
        price_model_path="models/price_model.pth"
    )
    
    # Run the loop. For testing, set interval to 5 seconds. In prod, set to 3600 (1h).
    orchestrator.start_bot(interval_seconds=3600)