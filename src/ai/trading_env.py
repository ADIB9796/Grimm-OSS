import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch

from src.risk.risk_manager import RiskManager
from src.models.price_predictor import PriceTransformer

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        super(TradingEnvironment, self).__init__()
        self.raw_data = data.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- CALCULATE RAW ATR FOR RISK MANAGER ---
        # (This adds an 11th column to raw_data)
        high_low = self.raw_data['high'] - self.raw_data['low']
        high_close = np.abs(self.raw_data['high'] - self.raw_data['close'].shift(1))
        low_close = np.abs(self.raw_data['low'] - self.raw_data['close'].shift(1))
        true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
        self.raw_data['atr'] = true_range.rolling(14).mean() 
        
        self.transaction_cost = transaction_cost
        
        # self.data will now have 15 columns total after this call
        self.data = self._prepare_features(self.raw_data)
        
        self.initial_balance = initial_balance
        self.risk_manager = RiskManager(balance=initial_balance)
        
        # --- TRANSFORMER SETUP ---
        self.seq_len = 50
        # FIX: Hardcode input_size=10 to perfectly match the trained checkpoint
        self.price_model = PriceTransformer(input_size=10).to(self.device)
        
        try:
            self.price_model.load_state_dict(torch.load("models/price_model.pth", map_location=self.device, weights_only=True))
        except FileNotFoundError:
            print("[WARNING] models/price_model.pth not found. Transformer is using random weights.")
            
        self.price_model.eval()
        self.history = []
        
        self.action_space = spaces.Discrete(3) 
        
        # RL Agent sees 15 Base Features + 3 Predictions = 18 total features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1] + 3,), dtype=np.float32
        )

    def _prepare_features(self, df):
        df = df.copy().select_dtypes(include=[np.number])
        # Adds 4 more columns, bringing total to 15
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["ma_fast"] = df["close"].rolling(5).mean()
        df["ma_slow"] = df["close"].rolling(20).mean()
        df["volatility"] = df["log_return"].rolling(10).std()
        df = df.dropna()

        # Normalize everything
        for col in df.columns:
            std = df[col].std()
            df[col] = 0 if std == 0 or np.isnan(std) else (df[col] - df[col].mean()) / (std + 1e-8)
                
        return df

    def get_prediction(self, seq):
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            pred = self.price_model(seq_tensor).cpu().numpy()[0]
        return pred

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.risk_manager.balance = self.initial_balance
        self.risk_manager.peak_balance = self.initial_balance
        
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.returns = []
        self.history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # full_state has 15 columns
        full_state = self.data.iloc[self.current_step].values.astype(np.float32)
        
        # FIX: Slice the first 10 columns for the Transformer's memory window
        transformer_input = full_state[:10]
        self.history.append(transformer_input)
        
        if len(self.history) > self.seq_len:
            self.history.pop(0)
            
        if len(self.history) == self.seq_len:
            pred = self.get_prediction(np.array(self.history))
        else:
            pred = np.array([0.0, 1.0, 0.0]) # Neutral
            
        # RL Agent gets all 15 features PLUS the 3 predictions
        return np.concatenate([full_state, pred]).astype(np.float32)

    def step(self, action):
        done = False
        offset = len(self.raw_data) - len(self.data)
        
        price = self.raw_data.iloc[self.current_step + offset]["close"]
        current_atr = self.raw_data.iloc[self.current_step + offset]["atr"]
        
        reward = 0
        trade_return = 0

        if action == 1: # BUY
            if self.position == 0:
                stop_loss = self.risk_manager.calculate_stop_loss(price, risk_percent=current_atr/price)
                size = self.risk_manager.calculate_position_size(price, stop_loss)
                
                if size > 0:
                    self.position = size 
                    self.entry_price = price
                    reward -= self.transaction_cost

        elif action == 2: # SELL
            if self.position > 0:
                trade_return = (price - self.entry_price) / self.entry_price
                pnl = self.position * (price - self.entry_price)
                self.risk_manager.update_balance(self.risk_manager.balance + pnl)
                
                self.position = 0
                reward -= self.transaction_cost

        self.returns.append(trade_return)
        
        if len(self.returns) > 10:
            mean = np.mean(self.returns)
            std = np.std(self.returns) + 1e-8
            reward += mean / std
        else:
            reward += trade_return

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            
        return self._get_observation(), reward, done, False, {}