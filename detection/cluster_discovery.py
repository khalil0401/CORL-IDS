"""
cluster_discovery.py

Module 8: Continual Class Discovery

Steps:
  1. Apply DBSCAN clustering to the buffered latent vectors z_t
  2. Filter out noise points (DBSCAN label == -1)
  3. Detect stable clusters (min_cluster_size)
  4. Assign new class IDs to stable clusters
  5. Trigger action space expansion in SAC agent + IDS environment
  6. Prevent catastrophic forgetting via EWC-lite (L2 stability penalty)

EWC-lite: we store a snapshot of parameters after each discovery event
and add an L2 regularisation term towards those anchors during future
network updates.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from typing import Optional, Dict, List


class ContinualClassDiscovery:
    """
    Manages new class detection and action-space expansion.

    Parameters
    ----------
    sac_agent      : DiscreteSAC instance
    env            : IDSEnvironment instance
    rarity_reward  : RarityReward instance  (updated when new classes appear)
    label_encoder  : sklearn LabelEncoder   (used for mapping)
    min_cluster_size : int  minimum samples for a cluster to be considered stable
    dbscan_eps     : float DBSCAN neighbourhood radius (latent space)
    dbscan_min_samples : int DBSCAN min_samples
    ewc_lambda     : float EWC regularisation strength
    """

    def __init__(self,
                 sac_agent,
                 env,
                 rarity_reward=None,
                 label_encoder=None,
                 min_cluster_size: int = 50,
                 dbscan_eps:        float = 1.5,
                 dbscan_min_samples: int = 30,
                 max_new_per_event: int = 3,
                 ewc_lambda:        float = 0.4):

        self.sac_agent         = sac_agent
        self.env               = env
        self.rarity_reward     = rarity_reward
        self.label_encoder     = label_encoder
        self.min_cluster_size  = min_cluster_size
        self.dbscan_eps        = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.max_new_per_event = max_new_per_event
        self.ewc_lambda        = ewc_lambda

        # Tracks next available class index
        self._next_class_id  = sac_agent.num_actions

        # EWC anchor: maps param_name → anchor_tensor
        self._ewc_anchors: Dict[str, torch.Tensor] = {}

        # History of discovered cluster centroids
        self._centroids: List[np.ndarray] = []

        # Count of total discovered classes
        self.num_discovered = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def discover(self, Z: np.ndarray, labels: np.ndarray) -> int:
        """
        Run clustering on unknown latents and expand action space if needed.

        Parameters
        ----------
        Z      : np.ndarray (N, latent_dim)
        labels : np.ndarray (N,)  original labels (-1 if truly unknown)

        Returns
        -------
        n_new : int  number of new classes integrated
        """
        if len(Z) < self.dbscan_min_samples:
            print(f"[DISCOVERY] Too few samples ({len(Z)}) for clustering.")
            return 0

        # ── 1. DBSCAN clustering ──────────────────────────────────────
        db = DBSCAN(eps=self.dbscan_eps,
                    min_samples=self.dbscan_min_samples,
                    n_jobs=-1)
        cluster_ids = db.fit_predict(Z)

        unique_clusters = [c for c in np.unique(cluster_ids) if c != -1]
        if not unique_clusters:
            print("[DISCOVERY] No stable clusters found.")
            return 0

        # ── 2. Filter stable clusters ─────────────────────────────────
        stable = []
        for c in unique_clusters:
            mask = cluster_ids == c
            if mask.sum() >= self.min_cluster_size:
                stable.append(c)

        if not stable:
            print(f"[DISCOVERY] {len(unique_clusters)} clusters found "
                  f"but none passed min_cluster_size={self.min_cluster_size}.")
            return 0

        n_new = len(stable)
        # Cap new classes per discovery event to prevent runaway expansion
        if n_new > self.max_new_per_event:
            print(f"[DISCOVERY] Capping {n_new} clusters to {self.max_new_per_event} (sorted by size).")
            # Keep the largest clusters
            stable_sizes = [(c, (cluster_ids == c).sum()) for c in stable]
            stable_sizes.sort(key=lambda x: x[1], reverse=True)
            stable = [c for c, _ in stable_sizes[:self.max_new_per_event]]
            n_new  = self.max_new_per_event

        print(f"[DISCOVERY] Found {n_new} stable new cluster(s) "
              f"from {len(Z)} unknown samples.")

        # ── 3. Compute centroids ──────────────────────────────────────
        new_centroids = []
        for c in stable:
            mask = cluster_ids == c
            centroid = Z[mask].mean(axis=0)
            new_centroids.append(centroid)
            self._centroids.append(centroid)

        # ── 4. Save EWC anchor BEFORE expanding ───────────────────────
        self._save_ewc_anchors()

        # ── 5. Expand action spaces ───────────────────────────────────
        new_total = self._next_class_id + n_new
        self.sac_agent.expand_action_space(new_total,
                                           lr=3e-4)
        self.env.expand_action_space(new_total)

        # ── 6. Update rarity reward with uniform probs for new classes ─
        if self.rarity_reward is not None:
            uniform_p = 1.0 / (new_total * 10)   # very rare initially
            new_probs = {i: uniform_p
                         for i in range(self._next_class_id, new_total)}
            self.rarity_reward.update_class_probs(new_probs)

        self._next_class_id  += n_new
        self.num_discovered  += n_new

        print(f"[DISCOVERY] Action space now: {new_total} classes "
              f"(discovered total: {self.num_discovered})")

        return n_new

    # ------------------------------------------------------------------
    # EWC-lite: stability regularisation
    # ------------------------------------------------------------------

    def _save_ewc_anchors(self):
        """Snapshot current actor+critic parameters as regularisation anchors."""
        self._ewc_anchors = {}
        for name, param in self.sac_agent.actor.named_parameters():
            self._ewc_anchors[f"actor.{name}"] = param.data.clone().detach()
        for name, param in self.sac_agent.q1.named_parameters():
            self._ewc_anchors[f"q1.{name}"] = param.data.clone().detach()
        for name, param in self.sac_agent.q2.named_parameters():
            self._ewc_anchors[f"q2.{name}"] = param.data.clone().detach()

    def ewc_penalty(self) -> torch.Tensor:
        """
        Compute L2 EWC-lite penalty term.

        Returns a scalar tensor to be added to the SAC losses.
        The penalty is zero if no anchors have been saved yet.
        """
        if not self._ewc_anchors:
            device = next(self.sac_agent.actor.parameters()).device
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0,
                                device=next(self.sac_agent.actor.parameters()).device)
        # Actor
        for name, param in self.sac_agent.actor.named_parameters():
            key = f"actor.{name}"
            if key in self._ewc_anchors:
                anchor = self._ewc_anchors[key].to(param.device)
                n = min(param.shape[0], anchor.shape[0])
                penalty += ((param[:n] - anchor[:n]) ** 2).sum()
        # Critics
        for net_name, net in [("q1", self.sac_agent.q1), ("q2", self.sac_agent.q2)]:
            for name, param in net.named_parameters():
                key = f"{net_name}.{name}"
                if key in self._ewc_anchors:
                    anchor = self._ewc_anchors[key].to(param.device)
                    n = min(param.shape[0], anchor.shape[0])
                    penalty += ((param[:n] - anchor[:n]) ** 2).sum()

        return self.ewc_lambda * penalty

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def get_centroids_2d(self) -> Optional[np.ndarray]:
        """Return 2-D PCA projection of cluster centroids for plotting."""
        if len(self._centroids) < 2:
            return None
        C = np.array(self._centroids)
        pca = PCA(n_components=2)
        return pca.fit_transform(C)
