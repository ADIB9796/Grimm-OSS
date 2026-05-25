import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import onnxruntime as ort

from src.risk.risk_manager import RiskManager

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=100.0, transaction_cost=0.0015, is_training=False):
        super(TradingEnvironment, self).__init__()
        self.raw_data = data.copy()
        
        # Calculate raw 1-hour ATR data for protective trailing risk bounds
        high_low = self.raw_data['high'] - self.raw_data['low']
        high_close = np.abs(self.raw_data['high'] - self.raw_data['close'].shift(1))
        low_close = np.abs(self.raw_data['low'] - self.raw_data['close'].shift(1))
        true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
        self.raw_data['atr'] = true_range.rolling(14).mean() 
        
        self.transaction_cost = transaction_cost
        self.data = self._prepare_features(self.raw_data)
        
        if len(self.data) == 0:
            raise ValueError("TradingEnvironment received empty dataset layout.")
        
        self.initial_balance = initial_balance
        self.risk_manager = RiskManager(
            balance=initial_balance, 
            leverage=400.0, # Target account leverage profile
            is_training=is_training
        )
        self.peak_balance = initial_balance 
        self.seq_len = 50
        
        # Inference Environment Initialization
        self.onnx_path = "models/BTC_price_model.onnx"
        try:
            self.price_model_sess = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
            self.input_name = self.price_model_sess.get_inputs()[0].name
        except Exception as e:
            print(f"[WARN] Transformer ONNX structural session missed at {self.onnx_path}. Details: {e}")
            self.price_model_sess = None
            
        # Pre-calculate predictions during initialization to maximize frame rates
        if self.price_model_sess is not None:
            print("      [INFO] Pre-computing ONNX Transformer array. Warming memory buffer...")
            self.precomputed_preds = self._precompute_all_predictions()
        else:
            self.precomputed_preds = [np.array([0.0, 1.0, 0.0], dtype=np.float32) for _ in range(len(self.data))]
            
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1] + 3,), dtype=np.float32
        )

    def _prepare_features(self, df):
        df = df.copy().select_dtypes(include=[np.number])
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["ma_fast"] = df["close"].rolling(5).mean()
        df["ma_slow"] = df["close"].rolling(20).mean()
        df["volatility_env"] = df["log_return"].rolling(10).std()
        df = df.dropna()

        for col in df.columns:
            std = df[col].std()
            df[col] = 0 if std == 0 or np.isnan(std) else (df[col] - df[col].mean()) / (std + 1e-8)
        return df

    def _precompute_all_predictions(self):
        predictions = []
        history = []
        for i in range(len(self.data)):
            full_state = self.data.iloc[i].values.astype(np.float32)
            transformer_input = full_state[:56]
            
            history.append(transformer_input)
            if len(history) > self.seq_len:
                history.pop(0)
                
            if len(history) == self.seq_len:
                seq_input = np.expand_dims(np.array(history), axis=0).astype(np.float32)
                logits = self.price_model_sess.run(None, {self.input_name: seq_input})[0][0]
                e_x = np.exp(logits - np.max(logits))
                probs = e_x / e_x.sum()
                predictions.append(probs)
            else:
                predictions.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return predictions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.risk_manager.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.returns = []
        return self._get_observation(), {}

    def _get_observation(self):
        full_state = self.data.iloc[self.current_step].values.astype(np.float32)
        current_pred = self.precomputed_preds[self.current_step]
        return np.concatenate([full_state, current_pred]).astype(np.float32)

    def step(self, action):
        done = False
        offset = len(self.raw_data) - len(self.data)
        price = self.raw_data.iloc[self.current_step + offset]["close"]
        
        # Real-world latency tracking: 3 Basis Points (0.03%) adverse slippage applied to execution nodes
        slippage_pct = 0.0003 
        raw_reward = 0
        trade_return = 0
        current_pred = self.precomputed_preds[self.current_step]

        if action == 1: # Entry Order execution
            if self.position == 0:
                win_prob = current_pred[2] 
                if win_prob >= 0.65: # Restrict to explicit breakout signals
                    margin_size = self.risk_manager.get_kelly_size(win_prob)
                    lot_size = self.risk_manager.calculate_lot_size(margin_size, price)
                    
                    if lot_size > 0:
                        # Convert fixed contract block (0.01 lot) to underlying asset weight
                        self.position = lot_size * 1.0 
                        self.entry_price = price * (1 + slippage_pct) # Buy higher due to latency
                        raw_reward -= self.transaction_cost
                else:
                    action = 0 

        elif action == 2: # Exit Order execution
            if self.position > 0:
                exit_price = price * (1 - slippage_pct) # Sell lower due to latency
                trade_return = (exit_price - self.entry_price) / self.entry_price
                
                pnl = self.position * (exit_price - self.entry_price)
                new_balance = self.risk_manager.balance + pnl
                self.risk_manager.update_performance(trade_return, new_balance)
                
                if self.risk_manager.balance > self.peak_balance:
                    self.peak_balance = self.risk_manager.balance
                
                self.position = 0
                raw_reward -= self.transaction_cost

        # CRITICAL $100 CAPITAL GUARD: Penalize aggressively if account swings lower than $4 (4%)
        current_drawdown = (self.peak_balance - self.risk_manager.balance) / self.peak_balance
        if current_drawdown > 0.04: 
            raw_reward -= (current_drawdown * 25.0) 
            
        if self.position > 0:
            unrealized_return = (price - self.entry_price) / self.entry_price
            if unrealized_return > 0:
                raw_reward += (unrealized_return * 0.2) 

        self.returns.append(trade_return)
        if len(self.returns) > 10:
            raw_reward += np.mean(self.returns) / (np.std(self.returns) + 1e-8)
        else:
            raw_reward += trade_return

        self.current_step += 1
        if self.current_step >= len(self.data) - 1 or self.risk_manager.balance <= 30.0:
            done = True # Ruin floor cutoff
            
        scaled_reward = np.sign(raw_reward) * np.log1p(np.abs(raw_reward))
        return self._get_observation(), scaled_reward, done, False, {}