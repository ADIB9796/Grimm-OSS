import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# =========================
# PRIORITIZED REPLAY BUFFER
# =========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha

    def add(self, experience):
        max_priority = max(self.priorities, default=1.0)

        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(max_priority)
        else:
            self.memory.pop(0)
            self.priorities.pop(0)
            self.memory.append(experience)
            self.priorities.append(max_priority)

    def sample(self, batch_size):
        priorities = np.array(self.priorities, dtype=np.float32)

        priorities = np.nan_to_num(priorities, nan=1e-6, posinf=1.0, neginf=1e-6)
        priorities = np.clip(priorities, 1e-6, None)

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        return samples, indices

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = float(abs(error) + 1e-6)

# =========================
# Q NETWORK WITH LAYER NORM
# =========================
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

    def forward(self, x):
        return self.network(x)

# =========================
# RL AGENT
# =========================
class RLAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon_decay=0.995, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.criterion = nn.MSELoss()
        self.memory = PrioritizedReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.update_target_every = 50
        self.step_count = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
            
        # TWEAK 4: ACTION THRESHOLD
        q_vals_array = q_values.cpu().numpy()[0]
        best_action = np.argmax(q_vals_array)
        
        # Action 0 is HOLD, Action 1 is BUY, Action 2 is SELL
        # The agent must have a Q-value advantage of at least 0.15 over "HOLD" to execute a trade.
        # Otherwise, it defaults to HOLD.
        if best_action in [1, 2]:
            if (q_vals_array[best_action] - q_vals_array[0]) < 0.15:
                return 0 
                
        return best_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory.memory) < self.batch_size:
            return

        samples, indices = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        next_actions = torch.argmax(self.model(next_states), dim=1)
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        errors = (current_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save(self, path="rl_trading_model.pth"):
        torch.save(self.model.state_dict(), path)