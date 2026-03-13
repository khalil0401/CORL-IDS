"""
data_loader.py

Loads the network traffic CSV, performs label detection,
splits into train/test sets, and computes class probabilities.

Supports a `hidden_classes` parameter:
    - Hidden classes are REMOVED from the training set
    - Hidden classes are KEPT in the test set (as ground truth unknowns)
    - This enables proper open-set evaluation:
        train on N-k classes, test whether the system discovers the k hidden ones
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional


# Columns that should never be treated as features (IPs, raw strings)
_DROP_COLS = {"src_ip", "dst_ip", "dns_query", "ssl_subject", "ssl_issuer",
              "http_uri", "http_user_agent", "http_orig_mime_types",
              "http_resp_mime_types", "weird_name", "weird_addl"}

# The target label column (attack type name)
LABEL_COL = "type"


def load_dataset(csv_path: str,
                 test_size: float = 0.2,
                 seed: int = 42,
                 hidden_classes: Optional[List[str]] = None):
    """
    Load dataset and return processed splits.

    Parameters
    ----------
    csv_path        : path to CSV file
    test_size       : fraction of data for test set
    seed            : random seed
    hidden_classes  : list of class name strings to HIDE from training.
                      These samples are excluded from X_train/y_train but
                      are included in X_test/y_test with their original labels.
                      Pass None (default) to use all classes.

    Returns
    -------
    X_train, X_test : pd.DataFrame  (raw, before scaling)
    y_train, y_test : np.ndarray    (integer encoded — same label space)
    label_encoder   : LabelEncoder  fitted on ALL labels (including hidden)
    class_probs     : dict[int, float]  empirical class probs on visible train classes
    feature_cols    : list[str]
    hidden_ids      : list[int]  label indices that were hidden (empty if none)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {csv_path}")

    # ------------------------------------------------------------------
    # Detect label column
    # ------------------------------------------------------------------
    if LABEL_COL not in df.columns:
        raise KeyError(f"Expected label column '{LABEL_COL}' not found. "
                       f"Available: {list(df.columns)}")

    df = df.dropna(subset=[LABEL_COL])

    # ------------------------------------------------------------------
    # Clean invalid labels (0, 1) from the raw TON_IoT_FULL
    # ------------------------------------------------------------------
    invalid_labels = {0, 1, "0", "1", " 0 ", " 1 ", "0.0", "1.0"}
    df = df[~df[LABEL_COL].isin(invalid_labels)]
    # Drop non-feature columns
    # ------------------------------------------------------------------
    drop = _DROP_COLS.union({"label"})
    feature_cols = [c for c in df.columns if c not in drop and c != LABEL_COL]

    # ------------------------------------------------------------------
    # Encode labels on the FULL dataset (so indices are consistent)
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y_all = le.fit_transform(df[LABEL_COL].astype(str).values)
    X_all = df[feature_cols].copy()

    # ------------------------------------------------------------------
    # Identify hidden class indices
    # ------------------------------------------------------------------
    hidden_classes = hidden_classes or []
    invalid = [c for c in hidden_classes if c not in le.classes_]
    if invalid:
        raise ValueError(f"Hidden classes not found in dataset: {invalid}\n"
                         f"Available: {list(le.classes_)}")

    hidden_ids = [int(le.transform([c])[0]) for c in hidden_classes]

    # ------------------------------------------------------------------
    # Stratified train/test split on ALL data
    # ------------------------------------------------------------------
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all,
        test_size=test_size,
        random_state=seed,
        stratify=y_all,
    )

    # ------------------------------------------------------------------
    # Remove hidden classes from TRAINING set only
    # ------------------------------------------------------------------
    if hidden_ids:
        visible_mask = ~np.isin(y_train_full, hidden_ids)
        X_train = X_train_full[visible_mask]
        y_train = y_train_full[visible_mask]
    else:
        X_train = X_train_full
        y_train = y_train_full

    # ------------------------------------------------------------------
    # Empirical class probabilities (on VISIBLE training classes only)
    # ------------------------------------------------------------------
    counts    = np.bincount(y_train, minlength=len(le.classes_))
    probs     = counts / max(counts.sum(), 1)
    class_probs = {i: max(p, 1e-8) for i, p in enumerate(probs)}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    visible_classes = [c for c in le.classes_ if c not in hidden_classes]
    print(f"All classes     ({len(le.classes_)}): {list(le.classes_)}")
    print(f"Visible (train) ({len(visible_classes)}): {visible_classes}")
    if hidden_classes:
        print(f"Hidden  (test)  ({len(hidden_classes)}): {hidden_classes}  "
              f"<-- will appear as UNKNOWN during evaluation")
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test, le, class_probs, feature_cols, hidden_ids


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "TON_IoT_Network_FULL.csv")
    # Example: hide 'ransomware' and 'ddos' from training
    load_dataset(path, hidden_classes=["ransomware", "ddos"])
