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


# Note: categorical_cols are now detected automatically by dtype (object/category)
# but can still be overridden manually if necessary.



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
        # Automatically detect categorical features based on dtypes
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"[PREPROCESS] Automatically detected categorical columns: {categorical_cols}")

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
    # 4. StandardScaler on numerical
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_enc.values).astype(np.float32)
    X_test_np  = scaler.transform(X_test_enc.values).astype(np.float32)

    feature_dim = X_train_np.shape[1]
    print(f"Feature dimension after preprocessing: {feature_dim}")

    return X_train_np, X_test_np, scaler, feature_dim
