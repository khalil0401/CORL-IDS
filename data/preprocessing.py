"""
preprocessing.py

Handles:
  - Removal of missing values
  - One-hot encoding of categorical columns
  - StandardScaler normalization of numerical columns
  - Returns fitted transformers for re-use at inference time
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Heuristic list of columns that often follow power-law distributions
# and benefit from log-scaling (case-insensitive check).
_LOG_PATTERNS = {"bytes", "pkts", "duration", "throughput", "bits", "size", "rate"}


# Note: categorical_cols are now detected automatically by dtype (object/category)
# but can still be overridden manually if necessary.



def is_categorical(series, threshold=50):
    """Check if a numeric series has low cardinality (likely categorical)."""
    return series.nunique() <= threshold

def preprocess(X_train: pd.DataFrame,
               X_test: pd.DataFrame,
               categorical_cols: list = None,
               ):
    """
    Fit preprocessing on train, transform both splits.

    Returns
    -------
    X_train_np, X_test_np : np.ndarray  (float32)
    scaler                : fitted StandardScaler
    feature_dim           : int
    """
    if categorical_cols is None:
        # Automatically detect categorical features based on dtypes AND cardinality
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Heuristic: columns with moderate unique values are likely categorical (e.g. Protocol, Flags, L7_Proto)
        for col in X_train.select_dtypes(include=['number']).columns:
            if is_categorical(X_train[col]):
                categorical_cols.append(col)
            
            # Special handling for Ports: Ports are categorical signals.
            # We can't one-hot encode all 65k, but we can encode the top-K.
            if "port" in col.lower():
                categorical_cols.append(col)
        
        categorical_cols = list(set(categorical_cols)) # Deduplicate
        print(f"[PREPROCESS] Automatically detected categorical columns: {categorical_cols}")

    # ------------------------------------------------------------------
    # 0.5. Port Bining (Top-K frequencies)
    # ------------------------------------------------------------------
    for col in X_train.columns:
        if "port" in col.lower():
            # Find top 30 most frequent ports in training
            top_ports = X_train[col].value_counts().index[:30].tolist()
            
            # Map values not in top-K to a placeholder string
            X_train[col] = X_train[col].apply(lambda x: str(x) if x in top_ports else "OTHER_PORT")
            X_test[col]  = X_test[col].apply(lambda x: str(x) if x in top_ports else "OTHER_PORT")
            
            # Ensure it's in categorical_cols for dummy encoding
            if col not in categorical_cols:
                categorical_cols.append(col)

    # ------------------------------------------------------------------
    # 1. Drop rows with NaN — already done at load; reset index here
    # ------------------------------------------------------------------
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Fill remaining NaN with 0 (some sparse fields like DNS columns)
    # ------------------------------------------------------------------
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # ------------------------------------------------------------------
    # 3. One-hot encode categorical columns
    #    fit on train, apply to both (unseen categories → 0)
    # ------------------------------------------------------------------
    num_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Cast categorical to string, numerical to float
    for c in categorical_cols:
        X_train[c] = X_train[c].astype(str)
        X_test[c]  = X_test[c].astype(str)
    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0)
        X_test[c]  = pd.to_numeric(X_test[c],  errors="coerce").fillna(0)

    # Get dummies from train, align test
    X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, dtype=float)
    X_test_enc  = pd.get_dummies(X_test,  columns=categorical_cols, dtype=float)

    # Align columns: test may miss some categories seen in train
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0.0)

    # ------------------------------------------------------------------
    # 3.5. Stabilize high-magnitude numerical features (Log-Scaling)
    # ------------------------------------------------------------------
    for col in X_train_enc.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in _LOG_PATTERNS):
            # Apply log(1+x) to dampen spikes and power-law distributions
            X_train_enc[col] = np.log1p(X_train_enc[col].abs())
            X_test_enc[col]  = np.log1p(X_test_enc[col].abs())

    # Failsafe: Remove any INF/NAN generated by outliers or log
    X_train_enc.replace([np.inf, -np.inf], 0, inplace=True)
    X_test_enc.replace([np.inf, -np.inf], 0, inplace=True)
    X_train_enc.fillna(0, inplace=True)
    X_test_enc.fillna(0, inplace=True)

    # ------------------------------------------------------------------
    # 4. StandardScaler on numerical
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_enc.values).astype(np.float32)
    X_test_np  = scaler.transform(X_test_enc.values).astype(np.float32)

    feature_dim = X_train_np.shape[1]
    print(f"Feature dimension after preprocessing: {feature_dim}")

    return X_train_np, X_test_np, scaler, feature_dim
