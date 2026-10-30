import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, cost, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        if isinstance(reward, np.ndarray):
            reward_scalar = reward.item()
        else:
            reward_scalar = float(reward)
        if isinstance(cost, np.ndarray):
            cost_scalar = cost.item()
        else:
            cost_scalar = float(cost)
        if isinstance(done, np.ndarray):
            done_scalar = done.item()
        else:
            done_scalar = float(done)
        transition = (state, action, reward_scalar, cost_scalar, next_state, done_scalar)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        actions = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        costs = np.array([b[3] for b in batch], dtype=np.float32)
        next_states = np.stack([b[4] for b in batch], axis=0).astype(np.float32)
        dones = np.array([b[5] for b in batch], dtype=np.float32)
        return (states, actions, rewards, costs, next_states, dones)
