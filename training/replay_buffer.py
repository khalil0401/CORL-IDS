"""
replay_buffer.py

Module 5: Experience Replay Buffer

Circular buffer storing transitions (state, action, reward, next_state, done).
Capacity: 100,000 transitions.
Supports random mini-batch sampling.
"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-size circular experience replay buffer.

    Parameters
    ----------
    capacity   : int   maximum number of transitions to store
    state_dim  : int   dimensionality of state vector
    """

    def __init__(self, capacity: int = 100_000, state_dim: int = 32, device: str = "cpu"):
        self.capacity  = capacity
        self.state_dim = state_dim
        self.device    = torch.device(device)

        self._states      = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self._actions     = torch.zeros((capacity,),           dtype=torch.int64,   device=self.device)
        self._rewards     = torch.zeros((capacity,),           dtype=torch.float32, device=self.device)
        self._next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self._dones       = torch.zeros((capacity,),           dtype=torch.float32, device=self.device)
        self._true_labels = torch.zeros((capacity,),           dtype=torch.int64,   device=self.device)

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

    def push_batch(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, true_labels: torch.Tensor=None):
        """Push a batch of transitions at once (vectorized)."""
        B = len(states)
        idxs = torch.arange(self._ptr, self._ptr + B, device=self.device) % self.capacity

        self._states[idxs]      = states
        self._actions[idxs]     = actions
        self._rewards[idxs]     = rewards
        self._next_states[idxs] = next_states
        self._dones[idxs]       = dones.to(torch.float32)
        if true_labels is not None:
            self._true_labels[idxs] = true_labels

        self._ptr  += B
        self._size  = min(self._size + B, self.capacity)

    # ------------------------------------------------------------------

    def sample(self, batch_size: int, class_weights: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch vectorized across device.
        """
        if self._size < batch_size:
            raise RuntimeError(
                f"Buffer has {self._size} transitions, need {batch_size}"
            )
        
        if class_weights is not None and hasattr(self, '_true_labels'):
            valid_labels = self._true_labels[:self._size]
            p = class_weights[valid_labels]
            # multinomial expects float probabilities and scales them automatically
            idx = torch.multinomial(p.to(torch.float32), batch_size, replacement=False)
        else:
            idx = torch.randperm(self._size, device=self.device)[:batch_size]
            
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
