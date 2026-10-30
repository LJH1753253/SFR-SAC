"""Pre-allocated ring replay buffer (float16 storage of states/actions, float32 scalars).

Same interface as utils.replay_buffer.ReplayBuffer: push(...) and sample(batch) ->
(states, actions, rewards, costs, next_states, dones) as float32 arrays. Memory per
transition ~ 2*(2*state_dim + action_dim) bytes, e.g. 4 KB for N=70, M=6.
"""
from __future__ import annotations

import numpy as np


class RingReplay:
    def __init__(self, capacity, state_dim, action_dim, seed=0, delivery_dim=0, energy_dim=0):
        self.capacity = int(capacity)
        self.s = np.zeros((self.capacity, state_dim), np.float16)
        self.s2 = np.zeros((self.capacity, state_dim), np.float16)
        self.a = np.zeros((self.capacity, action_dim), np.float16)
        self.r = np.zeros(self.capacity, np.float32)
        self.c = np.zeros(self.capacity, np.float32)
        self.d = np.zeros(self.capacity, np.float32)
        self.delivery = np.zeros((self.capacity, int(delivery_dim)), np.float32) if delivery_dim else None
        self.energy = np.zeros((self.capacity, int(energy_dim), 2), np.float32) if energy_dim else None
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def push(self, state, action, reward, cost, next_state, done, *, delivery_fraction=None, energy_components=None):
        if self.energy is not None:
            components = np.asarray(energy_components, dtype=np.float32)
            if components.shape != self.energy.shape[1:] or not np.isfinite(components).all() or np.any(components < 0):
                raise ValueError("Missing or invalid per-device energy labels")
        if self.delivery is not None:
            label = np.asarray(delivery_fraction, dtype=np.float32)
            if label.shape != self.delivery.shape[1:] or not np.isfinite(label).all():
                raise ValueError("Missing or invalid per-device delivery labels")
            if np.any(label < -1e-6) or np.any(label > 1.0 + 1e-6):
                raise ValueError("Delivery labels must be fractions of device workload")
        i = self.ptr
        self.s[i] = state
        self.a[i] = action
        self.r[i] = reward
        self.c[i] = cost
        self.s2[i] = next_state
        self.d[i] = float(done)
        if self.delivery is not None:
            self.delivery[i] = label
        if self.energy is not None:
            self.energy[i] = components
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return self._sample_indices(idx)

    def _sample_indices(self, idx):
        return (
            self.s[idx].astype(np.float32),
            self.a[idx].astype(np.float32),
            self.r[idx],
            self.c[idx],
            self.s2[idx].astype(np.float32),
            self.d[idx],
        )

    def sample_with_delivery(self, batch_size):
        if self.delivery is None:
            raise ValueError("Replay has no delivery labels")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return self._sample_indices(idx), self.delivery[idx].copy()

    def state_dict(self):
        return dict(capacity=self.capacity, size=self.size, ptr=self.ptr,
                    rng=self.rng.bit_generator.state,
                    delivery=None if self.delivery is None else self.delivery[:self.size],
                    energy=None if self.energy is None else self.energy[:self.size],
                    arrays={name: getattr(self, name)[:self.size]
                            for name in ("s", "s2", "a", "r", "c", "d")})

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity mismatch")
        self.size, self.ptr = state["size"], state["ptr"]
        energy = state.get("energy")
        if (energy is None) != (self.energy is None):
            raise ValueError("Replay energy-label configuration mismatch")
        if self.energy is not None:
            if energy.shape != (self.size, *self.energy.shape[1:]):
                raise ValueError("Replay energy-label shape mismatch")
            self.energy[:self.size] = energy
        labels = state.get("delivery")
        if (labels is None) != (self.delivery is None):
            raise ValueError("Replay delivery-label configuration mismatch")
        if self.delivery is not None:
            if labels.shape != (self.size, self.delivery.shape[1]):
                raise ValueError("Replay delivery-label shape mismatch")
            self.delivery[:self.size] = labels
        for name, values in state["arrays"].items():
            getattr(self, name)[:self.size] = values
        self.rng.bit_generator.state = state["rng"]

    def sample_with_physics(self, batch_size):
        if self.energy is None:
            raise ValueError("Replay has no energy labels")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return self._sample_indices(idx), None if self.delivery is None else self.delivery[idx].copy(), self.energy[idx].copy()
