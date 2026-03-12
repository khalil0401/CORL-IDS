"""
replay_buffer.py

Module 5: Experience Replay Buffer

Circular buffer storing transitions (state, action, reward, next_state, done).
Capacity: 100,000 transitions.
Supports random mini-batch sampling.
"""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-size circular experience replay buffer.

    Parameters
    ----------
    capacity   : int   maximum number of transitions to store
    state_dim  : int   dimensionality of state vector
    """

    def __init__(self, capacity: int = 100_000, state_dim: int = 32):
        self.capacity  = capacity
        self.state_dim = state_dim

        self._states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions     = np.zeros((capacity,),           dtype=np.int64)
        self._rewards     = np.zeros((capacity,),           dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones       = np.zeros((capacity,),           dtype=np.float32)
        self._true_labels = np.zeros((capacity,),           dtype=np.int64)

        self._ptr  = 0      # write pointer
        self._size = 0      # current fill level

    # ------------------------------------------------------------------

    def push(self,
             state:      np.ndarray,
             action:     int,
             reward:     float,
             next_state: np.ndarray,
             done:       bool):
        """Store a single transition."""
        idx = self._ptr % self.capacity

        self._states[idx]      = state
        self._actions[idx]     = action
        self._rewards[idx]     = reward
        self._next_states[idx] = next_state
        self._dones[idx]       = float(done)

        self._ptr  += 1
        self._size  = min(self._size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones, true_labels=None):
        """Push a batch of transitions at once (vectorized)."""
        B = len(states)
        idxs = np.arange(self._ptr, self._ptr + B) % self.capacity

        self._states[idxs]      = states
        self._actions[idxs]     = actions
        self._rewards[idxs]     = rewards
        self._next_states[idxs] = next_states
        self._dones[idxs]       = dones.astype(np.float32)
        if true_labels is not None:
            self._true_labels[idxs] = true_labels

        self._ptr  += B
        self._size  = min(self._size + B, self.capacity)

    # ------------------------------------------------------------------

    def sample(self, batch_size: int, class_weights: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """
        Sample a random mini-batch. 
        If class_weights is provided, performs balanced sampling based on true_labels.

        Returns
        -------
        states, actions, rewards, next_states, dones  — all np.ndarray
        """
        if self._size < batch_size:
            raise RuntimeError(
                f"Buffer has {self._size} transitions, need {batch_size}"
            )
        
        if class_weights is not None and hasattr(self, '_true_labels'):
            valid_labels = self._true_labels[:self._size]
            p = class_weights[valid_labels]
            p = p / p.sum()
            idx = np.random.choice(self._size, size=batch_size, replace=False, p=p)
        else:
            idx = np.random.choice(self._size, size=batch_size, replace=False)
            
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
        )

    # ------------------------------------------------------------------

    def __len__(self):
        return self._size

    @property
    def is_ready(self, min_size: int = None):
        return self._size > 0
