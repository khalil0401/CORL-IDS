"""
rarity_reward.py

Module 3: Rarity-Aware Reward Shaping

Formula:
    R_t = R_base * (1 + λ * log(1 + 1 / p(y_t)))

Where:
    R_base   = +1.0 (correct) | -1.0 (wrong)
    p(y_t)   = empirical probability of class y_t in training data
    λ        = rarity control coefficient (default 0.5)

Rare classes produce higher magnitude rewards, guiding the agent to
pay more attention to minority attack types.
"""

import math
import torch
from typing import Dict


class RarityReward:
    """
    Rarity-aware reward shaper.

    Parameters
    ----------
    class_probs : dict[int, float]  — empirical class probabilities from training
    lambda_     : float             — rarity scaling coefficient (default 0.5)
    """

    def __init__(self, class_probs: Dict[int, float], lambda_: float = 0.5):
        self.class_probs = class_probs
        self.lambda_     = lambda_

        # Precompute log-bonus for each class
        self._bonus_cache: Dict[int, float] = {}
        for cls, prob in class_probs.items():
            prob = max(prob, 1e-8)   # numerical safety
            self._bonus_cache[cls] = math.log(1.0 + 1.0 / prob)
            
        self._setup_bonus_tensor()

    def _setup_bonus_tensor(self):
        max_cls = max(self._bonus_cache.keys()) if self._bonus_cache else 0
        self._bonus_tensor = torch.full((max_cls + 1,), math.log(2.0), dtype=torch.float32)
        for cls, bonus in self._bonus_cache.items():
            self._bonus_tensor[cls] = bonus

    def compute(self, base_reward: float, true_label: int) -> float:
        """
        Apply rarity scaling to a base reward.

        Parameters
        ----------
        base_reward : +1.0 or -1.0
        true_label  : int  — ground-truth class index

        Returns
        -------
        scaled_reward : float
        """
        bonus = self._bonus_cache.get(true_label, math.log(2.0))
        # R_t = R_base * (1 + λ * log(1 + 1 / p(y_t)))
        # This properly scales both positive (+1) and negative (-1) rewards.
        # Mistakes on rare classes will issue larger negative penalties.
        return base_reward * (1.0 + self.lambda_ * bonus)

    def compute_batch(self, base_rewards: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """Apply rarity scaling vectorized across a batch."""
        self._bonus_tensor = self._bonus_tensor.to(base_rewards.device)
        bonuses = self._bonus_tensor[true_labels]
        return base_rewards * (1.0 + self.lambda_ * bonuses)

    def update_class_probs(self, new_probs: Dict[int, float]):
        """Refresh probabilities when new classes are discovered."""
        for cls, prob in new_probs.items():
            prob = max(prob, 1e-8)
            self.class_probs[cls] = prob
            self._bonus_cache[cls] = math.log(1.0 + 1.0 / (prob ** 0.5))
        self._setup_bonus_tensor()
