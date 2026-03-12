"""
sequence_builder.py

Converts a flat feature matrix into overlapping temporal sequences
using a sliding window of length seq_len.

Each output sample:  X[i] = features[i : i+seq_len]   shape (seq_len, feature_dim)
Corresponding label: y[i] = labels[i + seq_len - 1]   (label of the LAST step)
"""

import numpy as np


def build_sequences(X: np.ndarray,
                    y: np.ndarray,
                    seq_len: int = 10) -> tuple:
    """
    Parameters
    ----------
    X        : np.ndarray (N, feature_dim)  float32
    y        : np.ndarray (N,)              int
    seq_len  : int  sliding window length

    Returns
    -------
    X_seq : np.ndarray (M, seq_len, feature_dim)
    y_seq : np.ndarray (M,)
    where M = N - seq_len + 1
    """
    N, feat_dim = X.shape
    if N < seq_len:
        raise ValueError(f"Not enough samples ({N}) for seq_len={seq_len}")

    M = N - seq_len + 1
    # Efficient strided view
    X_seq = np.lib.stride_tricks.sliding_window_view(X, (seq_len, feat_dim))
    X_seq = X_seq.reshape(M, seq_len, feat_dim)          # (M, seq_len, feat_dim)
    y_seq = y[seq_len - 1:]                               # (M,)

    return X_seq.astype(np.float32), y_seq


def build_sequences_batched(X: np.ndarray,
                             y: np.ndarray,
                             seq_len: int = 10,
                             batch_size: int = 64):
    """
    Generator yielding (X_batch, y_batch) to avoid loading all sequences
    into memory at once for very large datasets.
    """
    X_seq, y_seq = build_sequences(X, y, seq_len)
    N = len(y_seq)
    indices = np.arange(N)
    np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        idx = indices[start: start + batch_size]
        yield X_seq[idx], y_seq[idx]
