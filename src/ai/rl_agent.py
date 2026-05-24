import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import random
import os

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
            nn.Linear(state_size, 512),
            nn.LayerNorm(512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x): 
        return self.network(x)

class RLAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon_decay=0.995, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer()
        
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.step_count = 0
        
        self.last_q_val = 0.0 # Tracking metric for stability monitoring

    def act(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.model(state_t).cpu().numpy()[0]
            
        # Log the max Q-value for training analysis
        self.last_q_val = float(np.max(q_vals))
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            
        action = int(np.argmax(q_vals))
        
        if action != 0 and (q_vals[action] - q_vals[0]) < 0.15:
            action = 0

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory.memory) < self.batch_size: 
            return
            
        samples, indices = self.memory.sample(self.batch_size)
        s, a, r, ns, d = zip(*samples)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

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

        self.memory.update_priorities(indices, (current_q - target_q).detach().cpu().numpy())
        self.step_count += 1
        if self.step_count % 50 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        
    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))

    def save_checkpoint(self, filepath, episode):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'episode': episode,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        return checkpoint['episode']
        
    def export_onnx(self, filepath):
        print(f"\n[INFO] Exporting RL Agent to ONNX at {filepath}...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.eval()
        self.model.to('cpu')
        
        dummy_input = torch.randn(1, self.state_size)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['state_input'],
            output_names=['q_values']
        )
        
        self.model.to(self.device)
        print("[SUCCESS] RL Agent exported to ONNX.")