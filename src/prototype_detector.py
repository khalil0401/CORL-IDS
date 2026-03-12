# -*- coding: utf-8 -*-
"""
prototype_detector.py  --  Prototype-Based Novelty Detector

Works independently of the policy entropy (so it's robust even when alpha->0).

Algorithm:
  BUILD (from training data):
    1. Encode all training sequences with the LSTM encoder -> z_t
    2. Compute per-class centroid: mu_c = mean(z_t | label=c)
    3. Compute per-class radius:   r_c  = mean ||z_t - mu_c||  (average intra-class dist)
    4. Threshold: tau_c = mu_dist_c + margin * std_dist_c

  DETECT (at inference):
    For each test sample z:
      d_c = ||z - mu_c||  for each known class c
      d_min = min(d_c)
      If d_min > tau_global  -> UNKNOWN
      Else                   -> KNOWN (classified by SAC policy)

Advantages over entropy-based detection:
  - Works when alpha -> 0 (policy is deterministic)
  - Geometrically meaningful: unknown attacks have different latent structure
  - Per-class adaptive thresholds
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class PrototypeDetector:
    """
    Detects samples as unknown when they are far from all known class prototypes.

    Parameters
    ----------
    margin : float
        Number of standard deviations beyond the mean intra-class distance
        to set the threshold. Higher = fewer unknowns flagged.
        Typical range: 1.0 - 3.0
    """

    def __init__(self, margin: float = 2.0):
        self.margin       = margin
        self.prototypes   : Optional[np.ndarray] = None  # (num_classes, latent_dim)
        self.threshold    : float = float("inf")
        self.class_radii  : Optional[np.ndarray] = None  # (num_classes,)
        self._fitted      = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self,
            Z: np.ndarray,
            y: np.ndarray,
            num_classes: Optional[int] = None) -> "PrototypeDetector":
        """
        Compute class prototypes and global threshold from labelled training data.

        Parameters
        ----------
        Z           : (N, latent_dim) encoded training samples
        y           : (N,)            integer class labels (contiguous 0..K-1)
        num_classes : optional override for K
        """
        K = num_classes or (int(y.max()) + 1)
        D = Z.shape[1]

        prototypes = np.zeros((K, D), dtype=np.float32)
        radii      = np.zeros(K, dtype=np.float32)
        radii_std  = np.zeros(K, dtype=np.float32)

        for c in range(K):
            mask = (y == c)
            if mask.sum() == 0:
                continue
            Zc        = Z[mask]
            mu_c      = Zc.mean(axis=0)
            dists     = np.linalg.norm(Zc - mu_c, axis=1)
            prototypes[c] = mu_c
            radii[c]      = dists.mean()
            radii_std[c]  = dists.std() + 1e-8

        self.prototypes  = prototypes
        self.class_radii = radii

        # Global threshold: accept a sample if its nearest-prototype distance
        # is within margin std-devs of that prototype's typical radius
        # tau_c = mean_radius_c + margin * std_radius_c
        # Global tau = max over all classes (conservative)
        self.threshold = float(np.max(radii + self.margin * radii_std))
        self._fitted   = True

        print(f"[PROTO] Prototypes fitted for {K} classes.")
        print(f"[PROTO] Per-class avg radius: "
              f"{', '.join(f'{c}:{r:.3f}' for c, r in enumerate(radii))}")
        print(f"[PROTO] Global threshold tau = {self.threshold:.4f}")
        return self

    # ------------------------------------------------------------------
    # Per-sample detection
    # ------------------------------------------------------------------

    def min_distance(self, z: np.ndarray) -> Tuple[float, int]:
        """
        Return (min_distance_to_prototype, nearest_class_id).
        z : (latent_dim,) single sample
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        dists = np.linalg.norm(self.prototypes - z, axis=1)
        idx   = int(np.argmin(dists))
        return float(dists[idx]), idx

    def is_unknown(self, z: np.ndarray) -> bool:
        """True if the sample is farther from all prototypes than the threshold."""
        d, _ = self.min_distance(z)
        return d > self.threshold

    def predict_batch(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        Z : (N, latent_dim)

        Returns
        -------
        unknown_mask  : (N,) bool   True = flagged as unknown
        min_dists     : (N,) float  distance to nearest prototype
        nearest_class : (N,) int    nearest known class index
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_batch().")
        dists_matrix = np.linalg.norm(
            Z[:, None, :] - self.prototypes[None, :, :], axis=2
        )                                          # (N, K)
        min_dists     = dists_matrix.min(axis=1)  # (N,)
        nearest_class = dists_matrix.argmin(axis=1)
        unknown_mask  = min_dists > self.threshold
        return unknown_mask, min_dists, nearest_class

    # ------------------------------------------------------------------
    # Threshold tuning helper
    # ------------------------------------------------------------------

    def set_margin(self, margin: float):
        """Re-compute threshold with a different margin (no re-fit needed)."""
        if not self._fitted:
            return
        radii     = self.class_radii
        # We stored radii mean; recompute conservatively
        self.margin    = margin
        self.threshold = float(np.max(radii) * (1.0 + margin * 0.3))
        print(f"[PROTO] Threshold updated -> tau = {self.threshold:.4f}  (margin={margin})")
