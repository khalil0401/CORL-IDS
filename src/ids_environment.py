"""
ids_environment.py

Module 2: Custom IDS Gym-like Environment

State  : z_t  ∈ R^32  (latent vector from LSTM encoder)
Actions: discrete attack class indices   {0, 1, ..., num_classes-1}
Reward : base ±1, optionally shaped by rarity reward module

Supports dynamic action space expansion (for continual class discovery).
"""

import numpy as np
import torch


class IDSEnvironment:
    """
    Gym-like environment for intrusion detection classification.

    Parameters
    ----------
    num_classes     : initial number of action classes
    rarity_reward   : RarityReward instance (or None for plain ±1)
    """

    CORRECT_REWARD   = 1.0
    INCORRECT_REWARD = -1.0

    def __init__(self, num_classes: int, rarity_reward=None):
        self.num_classes   = num_classes
        self.rarity_reward = rarity_reward

        # Current state (set via reset / step)
        self._state       = None
        self._true_label  = None
        self._done        = False

        # Current batch state (vectorized)
        self._batch_states = None
        self._batch_labels = None

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    def reset(self, state: np.ndarray, true_label: int):
        """
        Initialize environment with a new latent state.

        Parameters
        ----------
        state      : np.ndarray (latent_dim,)
        true_label : int  ground-truth class index
        """
        self._state      = state.astype(np.float32)
        self._true_label = int(true_label)
        self._done       = False
        return self._state.copy()

    def step(self, action: int):
        """
        Take a classification action.

        Returns
        -------
        next_state : same as current state (stateless env — IDS step)
        reward     : float  (rarity-shaped if rarity_reward is set)
        done       : bool   (always True; single-step episode)
        info       : dict
        """
        correct = (action == self._true_label)

        base_reward = self.CORRECT_REWARD if correct else self.INCORRECT_REWARD

        if self.rarity_reward is not None:
            reward = self.rarity_reward.compute(base_reward, self._true_label)
        else:
            reward = base_reward

        self._done = True
        info = {
            "correct":    correct,
            "true_label": self._true_label,
            "action":     action,
        }
        return self._state.copy(), reward, self._done, info

    def reset_batch(self, states: torch.Tensor, true_labels: torch.Tensor):
        """Vectorized reset for batch processing."""
        self._batch_states = states.clone()
        self._batch_labels = true_labels.clone()
        return self._batch_states

    def step_batch(self, actions: torch.Tensor):
        """Vectorized step for batch processing directly on GPU."""
        correct = (actions == self._batch_labels)
        
        base_rewards = torch.full_like(actions, self.INCORRECT_REWARD, dtype=torch.float32)
        base_rewards[correct] = self.CORRECT_REWARD
        
        if self.rarity_reward is not None:
            rewards = self.rarity_reward.compute_batch(base_rewards, self._batch_labels)
        else:
            rewards = base_rewards
            
        dones = torch.ones_like(rewards, dtype=torch.float32)
        return self._batch_states.clone(), rewards, dones, {}

    # ------------------------------------------------------------------
    # Continual learning: expand action space
    # ------------------------------------------------------------------

    def expand_action_space(self, new_num_classes: int):
        """Grow the number of actions to accommodate new discovered classes."""
        if new_num_classes <= self.num_classes:
            return
        print(f"[ENV] Expanding action space: {self.num_classes} -> {new_num_classes}")
        self.num_classes = new_num_classes

    @property
    def state(self):
        return self._state

    @property
    def action_dim(self):
        return self.num_classes
