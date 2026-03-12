"""
entropy_detector.py

Module 6: Entropy-Based Novelty Detection

Computes the Shannon entropy of the actor's policy distribution:
    H(π(·|s)) = −Σ_a π(a|s) log π(a|s)

Maintains running statistics (mean, std) via Welford's online algorithm
and uses an adaptive threshold to flag unknown/novel samples:
    τ = μ_H + β · σ_H     (β = 1.0 by default)

A sample is considered UNKNOWN if H > τ.
"""

import math
import numpy as np
import torch
from typing import Tuple


class EntropyDetector:
    """
    Adaptive entropy-based novelty detector.

    Parameters
    ----------
    beta       : float   threshold multiplier (default 1.0)
    min_samples: int     minimum samples before adaptive threshold activates
                         (uses a fixed high-entropy threshold until warmed up)
    """

    def __init__(self, beta: float = 1.0, min_samples: int = 200):
        self.beta        = beta
        self.min_samples = min_samples

        # Welford online mean/variance
        self._n    = 0
        self._mean = 0.0
        self._M2   = 0.0   # sum of squared deviations

    # ------------------------------------------------------------------

    def entropy_from_probs(self, probs: np.ndarray) -> float:
        """
        Compute Shannon entropy from a probability vector.

        Parameters
        ----------
        probs : np.ndarray (num_actions,)  must sum to ~1.0

        Returns
        -------
        H : float
        """
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def entropy_from_tensor(self, probs: torch.Tensor) -> np.ndarray:
        """
        Batch entropy computation from a probability tensor.

        Parameters
        ----------
        probs : Tensor (batch, num_actions)

        Returns
        -------
        H : np.ndarray (batch,)
        """
        probs_np = probs.detach().cpu().numpy()
        probs_np = np.clip(probs_np, 1e-10, 1.0)
        return -np.sum(probs_np * np.log(probs_np), axis=-1)

    # ------------------------------------------------------------------
    # Running statistics (Welford)
    # ------------------------------------------------------------------

    def update(self, H: float):
        """Update running mean and variance with a new entropy value."""
        self._n += 1
        delta     = H - self._mean
        self._mean += delta / self._n
        delta2    = H - self._mean
        self._M2  += delta * delta2

    def update_batch(self, H_arr: np.ndarray):
        """Batch update."""
        for h in H_arr:
            self.update(float(h))

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(self._M2 / (self._n - 1))

    @property
    def threshold(self) -> float:
        """Adaptive τ = μ + β·σ  (falls back to log(num_actions) if cold)."""
        return self._mean + self.beta * self.std

    @property
    def is_warm(self) -> bool:
        return self._n >= self.min_samples

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def is_unknown(self, H: float) -> bool:
        """
        Returns True if entropy H exceeds the adaptive threshold.
        Before warm-up, never flags unknown (conservative start).
        """
        if not self.is_warm:
            return False
        return H > self.threshold

    def detect_batch(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect unknown samples in a batch.

        Parameters
        ----------
        probs : np.ndarray (batch, num_actions)

        Returns
        -------
        H_arr    : np.ndarray (batch,)   per-sample entropy
        unknown  : np.ndarray (batch,)   bool mask of unknowns
        """
        probs = np.clip(probs, 1e-10, 1.0)
        H_arr = -np.sum(probs * np.log(probs), axis=-1)
        self.update_batch(H_arr)
        tau   = self.threshold
        unknown = (H_arr > tau) & self.is_warm
        return H_arr, unknown

    def stats(self) -> dict:
        return {
            "n":         self._n,
            "mean_H":   self._mean,
            "std_H":    self.std,
            "threshold": self.threshold,
            "is_warm":  self.is_warm,
        }
