import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch

from src.risk.risk_manager import RiskManager
from src.models.price_predictor import PriceTransformer

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.0025):
        super(TradingEnvironment, self).__init__()
        self.raw_data = data.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- CALCULATE RAW ATR FOR RISK MANAGER ---
        high_low = self.raw_data['high'] - self.raw_data['low']
        high_close = np.abs(self.raw_data['high'] - self.raw_data['close'].shift(1))
        low_close = np.abs(self.raw_data['low'] - self.raw_data['close'].shift(1))
        true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
        self.raw_data['atr'] = true_range.rolling(14).mean() 
        
        self.transaction_cost = transaction_cost
        self.data = self._prepare_features(self.raw_data)
        
        self.initial_balance = initial_balance
        self.risk_manager = RiskManager(balance=initial_balance)
        self.peak_balance = initial_balance 
        
        self.seq_len = 50
        self.price_model = PriceTransformer(input_size=10).to(self.device)
        
        try:
            self.price_model.load_state_dict(torch.load("models/price_model.pth", map_location=self.device, weights_only=True))
        except FileNotFoundError:
            pass
            
        self.price_model.eval()
        self.history = []
        self.current_pred = np.array([0.0, 1.0, 0.0])
        
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1] + 3,), dtype=np.float32
        )

    def _prepare_features(self, df):
        df = df.copy().select_dtypes(include=[np.number])
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["ma_fast"] = df["close"].rolling(5).mean()
        df["ma_slow"] = df["close"].rolling(20).mean()
        df["volatility"] = df["log_return"].rolling(10).std()
        df = df.dropna()

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
        self.peak_balance = self.initial_balance
        
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.returns = []
        self.history = []
        self.current_pred = np.array([0.0, 1.0, 0.0])
        
        return self._get_observation(), {}

    def _get_observation(self):
        full_state = self.data.iloc[self.current_step].values.astype(np.float32)
        transformer_input = full_state[:10] 
        
        self.history.append(transformer_input)
        if len(self.history) > self.seq_len:
            self.history.pop(0)
            
        if len(self.history) == self.seq_len:
            self.current_pred = self.get_prediction(np.array(self.history))
        else:
            self.current_pred = np.array([0.0, 1.0, 0.0])
            
        return np.concatenate([full_state, self.current_pred]).astype(np.float32)

    def step(self, action):
        done = False
        offset = len(self.raw_data) - len(self.data)
        
        price = self.raw_data.iloc[self.current_step + offset]["close"]
        current_atr = self.raw_data.iloc[self.current_step + offset]["atr"]
        
        reward = 0
        trade_return = 0

        # CRITICAL FIX: Index 2 represents UP class. Index 1 was NEUTRAL.
        transformer_up_confidence = self.current_pred[2]

        if action == 1: # BUY
            if self.position == 0:
                if transformer_up_confidence >= 0.65:
                    stop_loss = self.risk_manager.calculate_atr_stop_loss(price, current_atr)
                    size = self.risk_manager.calculate_position_size(price, stop_loss)
                    
                    if size > 0:
                        self.position = size 
                        self.entry_price = price
                        reward -= self.transaction_cost
                else:
                    action = 0 # VETO

        elif action == 2: # SELL
            if self.position > 0:
                trade_return = (price - self.entry_price) / self.entry_price
                pnl = self.position * (price - self.entry_price)
                self.risk_manager.update_balance(self.risk_manager.balance + pnl)
                
                if self.risk_manager.balance > self.peak_balance:
                    self.peak_balance = self.risk_manager.balance
                
                self.position = 0
                reward -= self.transaction_cost

        # REWARD SHAPING
        current_drawdown = (self.peak_balance - self.risk_manager.balance) / self.peak_balance
        if current_drawdown > 0.02: 
            reward -= (current_drawdown * 15) 
            
        if self.position > 0:
            unrealized_return = (price - self.entry_price) / self.entry_price
            if unrealized_return > 0:
                reward += (unrealized_return * 0.1) 

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