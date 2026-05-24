import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import onnxruntime as ort

from src.risk.risk_manager import RiskManager

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.0015, is_training=False):
        super(TradingEnvironment, self).__init__()
        self.raw_data = data.copy()
        
        # --- CALCULATE RAW ATR FOR RISK MANAGER ---
        high_low = self.raw_data['high'] - self.raw_data['low']
        high_close = np.abs(self.raw_data['high'] - self.raw_data['close'].shift(1))
        low_close = np.abs(self.raw_data['low'] - self.raw_data['close'].shift(1))
        true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
        self.raw_data['atr'] = true_range.rolling(14).mean() 
        
        self.transaction_cost = transaction_cost
        
        # Prepare features adds extra context indicators to the end of the 56 base columns
        self.data = self._prepare_features(self.raw_data)
        
        if len(self.data) == 0:
            raise ValueError("TradingEnvironment received empty data.")
        
        self.initial_balance = initial_balance
        self.risk_manager = RiskManager(
            balance=initial_balance, 
            leverage=50.0, 
            is_training=is_training
        )
        self.peak_balance = initial_balance 
        self.seq_len = 50
        
        # CRITICAL UPDATE: Load Transformer using ONNX Runtime
        self.onnx_path = "models/BTC_price_model.onnx"
        try:
            self.price_model_sess = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
            self.input_name = self.price_model_sess.get_inputs()[0].name
        except Exception as e:
            print(f"[WARN] Transformer ONNX not found at {self.onnx_path}. Error: {e}")
            self.price_model_sess = None
            
        # --- CRITICAL SPEED UPDATE: PRE-COMPUTE ALL PREDICTIONS ---
        if self.price_model_sess is not None:
            print("      [INFO] Pre-computing ONNX Transformer predictions. This takes ~5 seconds once...")
            self.precomputed_preds = self._precompute_all_predictions()
        else:
            self.precomputed_preds = [np.array([0.0, 1.0, 0.0], dtype=np.float32) for _ in range(len(self.data))]
            
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
        df["volatility_env"] = df["log_return"].rolling(10).std()
        df = df.dropna()

        for col in df.columns:
            std = df[col].std()
            df[col] = 0 if std == 0 or np.isnan(std) else (df[col] - df[col].mean()) / (std + 1e-8)
                
        return df

    def _precompute_all_predictions(self):
        """Runs the sliding window over the static dataset ONE TIME to save hours of training."""
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
        # O(1) Instant Lookup instead of ONNX Inference
        full_state = self.data.iloc[self.current_step].values.astype(np.float32)
        current_pred = self.precomputed_preds[self.current_step]
            
        return np.concatenate([full_state, current_pred]).astype(np.float32)

    def step(self, action):
        done = False
        offset = len(self.raw_data) - len(self.data)
        
        price = self.raw_data.iloc[self.current_step + offset]["close"]
        
        # LIVE SIMULATION: 3 Basis Points (0.03%) adverse slippage for execution latency
        slippage_pct = 0.0003 
        
        raw_reward = 0
        trade_return = 0

        # Look up pre-computed prediction
        current_pred = self.precomputed_preds[self.current_step]

        if action == 1: 
            if self.position == 0:
                win_prob = current_pred[2] 
                if win_prob >= 0.60:
                    size = self.risk_manager.get_kelly_size(win_prob)
                    if size > 0:
                        self.position = size 
                        # Adverse Execution: You buy higher than expected
                        self.entry_price = price * (1 + slippage_pct)
                        raw_reward -= self.transaction_cost
                else:
                    action = 0 

        elif action == 2: 
            if self.position > 0:
                # Adverse Execution: You sell lower than expected
                exit_price = price * (1 - slippage_pct)
                
                trade_return = (exit_price - self.entry_price) / self.entry_price
                pnl = self.position * (exit_price - self.entry_price)
                new_balance = self.risk_manager.balance + pnl
                self.risk_manager.update_performance(trade_return, new_balance)
                
                if self.risk_manager.balance > self.peak_balance:
                    self.peak_balance = self.risk_manager.balance
                
                self.position = 0
                raw_reward -= self.transaction_cost
                
        current_drawdown = (self.peak_balance - self.risk_manager.balance) / self.peak_balance
        if current_drawdown > 0.02: 
            raw_reward -= (current_drawdown * 10) 
            
        if self.position > 0:
            unrealized_return = (price - self.entry_price) / self.entry_price
            if unrealized_return > 0:
                raw_reward += (unrealized_return * 0.2) 

        self.returns.append(trade_return)
        
        if len(self.returns) > 10:
            mean = np.mean(self.returns)
            std = np.std(self.returns) + 1e-8
            raw_reward += mean / std
        else:
            raw_reward += trade_return

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            
        scaled_reward = np.sign(raw_reward) * np.log1p(np.abs(raw_reward))
            
        return self._get_observation(), scaled_reward, done, False, {}