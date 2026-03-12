"""
unknown_buffer.py

Module 7: Unknown Sample Buffer

Stores latent vectors z_t that the entropy detector has flagged as novel/unknown.
Alongside each vector, stores the true label (if available) and a timestamp.

When the buffer reaches a configurable threshold, it triggers the
cluster_discovery module to process the accumulated unknown samples.
"""

import time
import numpy as np
from typing import Optional


class UnknownBuffer:
    """
    Memory buffer for unknown (high-entropy) samples.

    Parameters
    ----------
    trigger_size : int   number of unknown samples that triggers clustering
    max_size     : int   maximum buffer size (old samples discarded after this)
    """

    def __init__(self, trigger_size: int = 200, max_size: int = 5000):
        self.trigger_size = trigger_size
        self.max_size     = max_size

        self._latents    : list[np.ndarray] = []   # z_t  (latent_dim,)
        self._labels     : list[int]        = []   # true label (may be -1 if unknown)
        self._timestamps : list[float]      = []   # unix timestamp

        self._trigger_count = 0   # how many times clustering was triggered

    # ------------------------------------------------------------------

    def add(self, z: np.ndarray, true_label: int = -1):
        """
        Add a flagged unknown sample.

        Parameters
        ----------
        z          : np.ndarray (latent_dim,)
        true_label : int  (original label index; -1 if truly unknown)
        """
        if len(self._latents) >= self.max_size:
            # Drop oldest half to make room
            half = self.max_size // 2
            self._latents    = self._latents[half:]
            self._labels     = self._labels[half:]
            self._timestamps = self._timestamps[half:]

        self._latents.append(z.copy())
        self._labels.append(int(true_label))
        self._timestamps.append(time.time())

    def add_batch(self, Z: np.ndarray, labels: Optional[np.ndarray] = None):
        """Add multiple unknown samples at once."""
        if labels is None:
            labels = np.full(len(Z), -1, dtype=int)
        for z, lbl in zip(Z, labels):
            self.add(z, int(lbl))

    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._latents)

    @property
    def should_trigger(self) -> bool:
        return self.size >= self.trigger_size

    def peek(self) -> tuple:
        """Return (latents, labels, timestamps) without clearing."""
        if not self._latents:
            return np.empty((0,)), np.empty((0,), int), []
        return (
            np.array(self._latents, dtype=np.float32),
            np.array(self._labels,  dtype=np.int64),
            list(self._timestamps),
        )

    def flush(self) -> tuple:
        """Return all stored data and clear the buffer."""
        Z_arr, y_arr, ts = self.peek()
        self._latents    = []
        self._labels     = []
        self._timestamps = []
        self._trigger_count += 1
        return Z_arr, y_arr, ts

    def stats(self) -> dict:
        return {
            "size":           self.size,
            "trigger_size":   self.trigger_size,
            "trigger_count":  self._trigger_count,
            "should_trigger": self.should_trigger,
        }
