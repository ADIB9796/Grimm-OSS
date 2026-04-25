import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .risk_manager import RiskManager

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha

    def add(self, experience):
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
            self.priorities.pop(0)
        self.memory.append(experience)
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        priorities = np.clip(np.array(self.priorities, dtype=np.float32), 1e-6, None)
        probs = (priorities ** self.alpha) / (priorities ** self.alpha).sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        return [self.memory[i] for i in indices], indices

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = float(abs(error) + 1e-6)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x): return self.network(x)

class RLAgent:
    def __init__(self, state_size, action_size, initial_balance=10000, lr=1e-4, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core Models
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Risk & Speed
        self.risk_manager = RiskManager(balance=initial_balance)
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer()
        self.gamma, self.epsilon, self.epsilon_decay = gamma, 1.0, 0.995
        self.batch_size, self.step_count = 64, 0

    def act(self, state, transformer_confidence=0.5):
        """Returns (action, position_size) based on Q-values and Kelly math."""
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.model(state_t).cpu().numpy()[0]
            action = np.argmax(q_vals)
            # Use Q-value advantage check (from your TWEAK 4)
            if action != 0 and (q_vals[action] - q_vals[0]) < 0.15:
                action = 0

        # Calculate sizing for non-hold actions
        size = self.risk_manager.get_kelly_size(transformer_confidence) if action != 0 else 0.0
        return action, size

    def replay(self):
        if len(self.memory.memory) < self.batch_size: return
        samples, indices = self.memory.sample(self.batch_size)
        s, a, r, ns, d = zip(*samples)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        # Mixed Precision Training
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            current_q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
            next_actions = torch.argmax(self.model(ns), dim=1)
            next_q = self.target_model(ns).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = r + (1 - d) * self.gamma * next_q
            loss = nn.MSELoss()(current_q, target_q.detach())

        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Update Buffer Priorities
        self.memory.update_priorities(indices, (current_q - target_q).detach().cpu().numpy())
        self.step_count += 1
        if self.step_count % 50 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)