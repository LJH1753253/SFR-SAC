"""Uniform/recent replay mixture with aligned physical labels."""

import numpy as np

from v2.replay import RingReplay


class RecencyReplay(RingReplay):
    def __init__(self, *args, recent_fraction=0., recent_window=20000, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0. <= recent_fraction <= 1. or not np.isfinite(recent_fraction):
            raise ValueError("recent_fraction must be in [0, 1]")
        if int(recent_window) != recent_window or recent_window < 1:
            raise ValueError("recent_window must be a positive integer")
        self.recent_fraction = float(recent_fraction)
        self.recent_window = int(recent_window)
        self.last_sample_stats = {}

    def draw_indices(self, batch_size):
        if not self.size:
            raise ValueError("Cannot sample an empty replay")
        batch_size = int(batch_size)
        recent_count = int(batch_size*self.recent_fraction)
        uniform = self.rng.integers(0, self.size, size=batch_size-recent_count)
        if recent_count:
            ages = self.rng.integers(0, min(self.size, self.recent_window), size=recent_count)
            recent = (self.ptr - 1 - ages) % self.capacity
            idx = np.concatenate((uniform, recent))
        else:
            idx = uniform
        age = (self.ptr - 1 - idx) % self.capacity
        self.last_sample_stats = dict(mean_age_steps=float(age.mean()),
            recent_fraction=float(np.mean(age < self.recent_window)),
            forced_recent_fraction=recent_count/max(batch_size, 1))
        return idx

    def sample(self, batch_size):
        return self._sample_indices(self.draw_indices(batch_size))

    def sample_with_delivery(self, batch_size):
        if self.delivery is None:
            raise ValueError("Replay has no delivery labels")
        idx = self.draw_indices(batch_size)
        return self._sample_indices(idx), self.delivery[idx].copy()

    def sample_with_physics(self, batch_size):
        if self.energy is None:
            raise ValueError("Replay has no energy labels")
        idx = self.draw_indices(batch_size)
        return self._sample_indices(idx), None if self.delivery is None else self.delivery[idx].copy(), self.energy[idx].copy()

    def state_dict(self):
        state = super().state_dict()
        state["sampling"] = dict(recent_fraction=self.recent_fraction, recent_window=self.recent_window)
        return state

    def load_state_dict(self, state):
        expected = dict(recent_fraction=self.recent_fraction, recent_window=self.recent_window)
        if "sampling" in state and state["sampling"] != expected:
            raise ValueError("Replay sampling configuration mismatch")
        super().load_state_dict(state)
