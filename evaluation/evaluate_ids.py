# -*- coding: utf-8 -*-
"""
evaluate.py  --  CORL-IDS Evaluation + Open-Set Discovery

Usage:
    python src/evaluate.py [--model models/trained_model.pt]

Two-phase evaluation:
  PHASE 1 - Known-class classification
    * Load trained model (knows 10 classes)
    * Run inference on test set
    * Report: accuracy, precision, recall, F1, False Alarm Rate
    * Detect high-entropy samples -> flag as "UNKNOWN"

  PHASE 2 - Continual class discovery  (open-set)
    * Collect all flagged unknown latent vectors
    * Run DBSCAN clustering
    * Assign new class IDs to stable clusters
    * Expand the SAC action space
    * Report: unknown attack detection rate, cluster summary
    * Save confusion matrix + entropy plots
"""

import os
import sys
import argparse
import pickle
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.dataset_loader import load_dataset
from data.preprocessing import preprocess
from data.sequence_builder import build_sequences
from models.lstm_encoder import LSTMEncoder
from models.sac_agent import DiscreteSAC
from detection.confidence_unknown import ConfidenceUnknownDetector
from detection.centroid_detector import CentroidDetector
from training.unknown_buffer import UnknownBuffer
from detection.cluster_discovery import ContinualClassDiscovery
from training.ids_environment import IDSEnvironment


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR   = os.path.join(BASE_DIR, "logs")


# =======================================================================
# Checkpoint helpers
# =======================================================================

def load_checkpoint(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def load_sklearn(ckpt_path):
    pkl = ckpt_path.replace(".pt", "_sklearn.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Sklearn pickle not found: {pkl}")
    with open(pkl, "rb") as f:
        return pickle.load(f)


# =======================================================================
# PHASE 1 - Known-class inference
# =======================================================================

def run_inference(encoder, sac, X_seq, y_seq, device, batch_size=512):
    """
    Run forward pass over test sequences using vectorized batch operations.
    """
    N = len(y_seq)
    all_preds = []
    all_true  = []
    all_probs = []
    all_z     = []

    encoder.eval()
    sac.actor.eval()

    with torch.no_grad():
        for start in range(0, N, batch_size):
            xb = torch.tensor(X_seq[start: start + batch_size],
                               dtype=torch.float32, device=device)
            yb = y_seq[start: start + batch_size]

            z_batch = encoder(xb)
            actions, probs = sac.select_action_batch(z_batch, deterministic=True)

            all_preds.extend(actions.cpu().tolist())
            all_true.extend(yb.tolist())
            all_probs.append(probs.cpu().numpy())
            all_z.append(z_batch.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_true),
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_z, axis=0))


# =======================================================================
# Main evaluation function
# =======================================================================

def evaluate(ckpt_path, device_str="cpu",
             dbscan_eps=1.2, dbscan_min_samples=30,
             min_cluster_size=50, beta_entropy=1.0,
             proto_margin=2.0):

    device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"  CORL-IDS Evaluation  device={device}")
    print(f"{'='*60}\n")

    # -- Load checkpoint -------------------------------------------------
    ckpt   = load_checkpoint(ckpt_path, device)
    sk_obj = load_sklearn(ckpt_path)
    scaler = sk_obj["scaler"]
    le     = sk_obj["le"]
    cfg    = ckpt["cfg"]
    meta   = ckpt["meta"]

    num_known     = meta["num_classes"]     # number of visible (trained) classes
    feature_dim   = meta["feature_dim"]
    latent_dim    = cfg["latent_dim"]
    lstm_hidden   = cfg["lstm_hidden"]
    seq_len       = cfg["seq_len"]
    visible_names = meta.get("visible_names", meta["le_classes"])
    visible_ids   = meta.get("visible_ids",   list(range(num_known)))
    hidden_names  = meta.get("hidden_names",  [])
    hidden_ids    = meta.get("hidden_ids",    [])

    print(f"Model info:")
    print(f"  Epochs trained  : {meta['epoch']}")
    print(f"  Visible classes : {num_known}  {visible_names}")
    if hidden_names:
        print(f"  Hidden classes  : {hidden_names}  <-- model has never seen these")
    print()

    # -- Rebuild networks ------------------------------------------------
    encoder = LSTMEncoder(feature_dim, lstm_hidden, latent_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])

    sac = DiscreteSAC(
        state_dim   = latent_dim,
        num_actions = num_known,
        device      = device_str,
        auto_alpha  = False,
        alpha       = cfg.get("alpha_entropy", 0.2),
    )
    sac.load_state_dict_full(ckpt["sac"])

    env = IDSEnvironment(num_known)

    # Build remapping: original le index -> visible (SAC) index
    # Needed so we can compare predictions against the test set labels
    orig_to_vis = {orig: vis for vis, orig in enumerate(visible_ids)}
    vis_to_orig = {vis: orig for orig, vis in orig_to_vis.items()}

    # -- Load test data --------------------------------------------------
    # IMPORTANT: pass same hidden_classes so preprocessing (scaler, one-hot)
    # matches what was used during training (same feature_dim).
    csv_path = os.path.join(BASE_DIR, "train_test_network.csv")
    X_train_raw, X_test_raw, y_train, y_test, le, _, _, _ = load_dataset(
        csv_path,
        seed=cfg.get("seed", 42),
        hidden_classes=hidden_names,   # same as training
    )
    X_train_np, X_test_np, _, _ = preprocess(X_train_raw, X_test_raw)
    X_seq, y_seq = build_sequences(X_test_np, y_test, seq_len)
    print(f"Test sequences : {len(y_seq):,}")
    print(f"  (includes hidden class samples for open-set testing)")

    # y_seq contains ORIGINAL label indices (including hidden classes)
    # For Phase 1 metrics we only evaluate samples whose true class is VISIBLE
    # (hidden class samples will naturally get high entropy -> flagged unknown)
    is_visible_sample = np.isin(y_seq, visible_ids)
    is_hidden_sample  = np.isin(y_seq, hidden_ids)

    # Remap visible true labels to SAC indices for metric computation
    y_seq_vis = np.array(
        [orig_to_vis.get(int(lbl), -1) for lbl in y_seq], dtype=np.int64
    )

    # Prime Confidence detector (Highly aggressive)
    conf_det = ConfidenceUnknownDetector(threshold=0.98)

    # -- Fit Centroid (Mahalanobis) Detector from TRAINING data -----------------------
    print("\nFitting Centroid/Mahalanobis detector on training data ...")
    X_train_np_full, _, _, _, _, _, _, _ = load_dataset(
        csv_path, seed=cfg.get("seed", 42), hidden_classes=hidden_names
    )
    # Reuse same preprocessor (scaler fitted on training data)
    X_train_np_arr, _, _, _ = preprocess(X_train_raw, X_test_raw)
    
    # We can process the entire training set for centroids natively now
    X_seq_tr, y_seq_tr = build_sequences(X_train_np_arr, y_train, seq_len)
    
    y_seq_tr_vis = np.array(
        [orig_to_vis.get(int(lbl), -1) for lbl in y_seq_tr], dtype=np.int64
    )
    valid_tr = y_seq_tr_vis >= 0
    X_seq_tr  = X_seq_tr[valid_tr]
    y_seq_tr_vis = y_seq_tr_vis[valid_tr]

    encoder.eval()
    with torch.no_grad():
        Z_train_list = []
        bs = 512
        for si in range(0, len(X_seq_tr), bs):
            xb = torch.tensor(X_seq_tr[si:si+bs], dtype=torch.float32, device=device)
            Z_train_list.append(encoder(xb).cpu().numpy())
    Z_train_full = np.concatenate(Z_train_list, axis=0)

    centroid_det = CentroidDetector(distance_multiplier=1.0)
    centroid_det.fit(Z_train_full, y_seq_tr_vis, num_classes=num_known)

    # ===================================================================
    # PHASE 1 - Known-class inference
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1 - Known-class classification")
    print(f"{'='*60}")

    all_preds, all_true, all_probs, all_z = run_inference(
        encoder, sac, X_seq, y_seq, device
    )

    # Flag unknowns: confidence bounds OR centroid bounds
    centroid_unknown = centroid_det.predict_batch(all_z)
    conf_unknown = conf_det.predict_batch(all_probs)

    # Combined: a sample is UNKNOWN if either detector flags it
    unknown_mask = centroid_unknown | conf_unknown
    known_mask   = ~unknown_mask

    # Breakdown: how many hidden-class samples were flagged?
    hidden_flagged = (unknown_mask & is_hidden_sample).sum()
    hidden_total   = is_hidden_sample.sum()

    print(f"\n  Confidence threshold (tau_conf)  : {conf_det.threshold:.2f}")
    print(f"  Flagged by representation distance (Centroid) : {centroid_unknown.sum():,}")
    print(f"  Flagged by absolute policy confidence         : {conf_unknown.sum():,}")
    print(f"  Flagged by EITHER (union)        : {unknown_mask.sum():,} / {len(all_preds):,} "
          f"({100*unknown_mask.mean():.1f}%)")
    if hidden_total > 0:
        print(f"  Hidden-class samples : {hidden_total:,} in test set")
        print(f"  Hidden flagged UNKNOWN : {hidden_flagged:,} / {hidden_total:,} "
              f"({100*hidden_flagged/hidden_total:.1f}%)  <-- goal: HIGH")

    # Compute Phase 1 metrics on VISIBLE-class samples NOT flagged as unknown
    vis_known_mask  = known_mask & is_visible_sample
    if vis_known_mask.sum() > 0:
        y_true_eval = y_seq_vis[vis_known_mask]   # SAC indices
        y_pred_eval = all_preds[vis_known_mask]   # SAC indices

        acc  = accuracy_score(y_true_eval, y_pred_eval)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_eval, y_pred_eval, average="macro", zero_division=0
        )

        # False Alarm Rate using normal/benign class
        normal_name = "normal" if "normal" in visible_names \
                      else ("benign" if "benign" in visible_names else None)
        if normal_name:
            normal_vis_idx = orig_to_vis[int(le.transform([normal_name])[0])]
            y_bin_true = (y_true_eval != normal_vis_idx).astype(int)
            y_bin_pred = (y_pred_eval != normal_vis_idx).astype(int)
            FP  = ((y_bin_true == 0) & (y_bin_pred == 1)).sum()
            TN  = ((y_bin_true == 0) & (y_bin_pred == 0)).sum()
            far = FP / (FP + TN + 1e-8)
        else:
            far = float("nan")

        SEP = "-" * 60
        print(f"\n{SEP}")
        print(f"  Accuracy  (visible, non-flagged) : {acc*100:.2f}%")
        print(f"  Macro Precision                  : {prec*100:.2f}%")
        print(f"  Macro Recall                     : {rec*100:.2f}%")
        print(f"  Macro F1-Score                   : {f1*100:.2f}%")
        print(f"  False Alarm Rate                 : {far*100:.4f}%"
              if not math.isnan(far) else
              f"  False Alarm Rate                 : N/A")
        print(SEP)

        labels_in   = sorted(set(y_true_eval.tolist()) | set(y_pred_eval.tolist()))
        target_names = [visible_names[l] for l in labels_in if l < len(visible_names)]
        labels_in    = [l for l in labels_in if l < len(visible_names)]
        report = classification_report(
            y_true_eval, y_pred_eval,
            labels=labels_in,
            target_names=target_names,
            zero_division=0,
        )
        print(f"\nClassification Report (visible classes only):\n")
        print(report)

        # Confusion matrix (visible classes only)
        os.makedirs(LOG_DIR, exist_ok=True)
        cm = confusion_matrix(y_true_eval, y_pred_eval, labels=labels_in)
        fig, ax = plt.subplots(figsize=(max(8, len(labels_in)),
                                        max(6, len(labels_in) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names,
                    yticklabels=target_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (visible classes, {num_known} classes trained)")
        plt.tight_layout()
        cm_path = os.path.join(LOG_DIR, "confusion_matrix.png")
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        print(f"  [OK] Confusion matrix -> {cm_path}")

    acc  = acc  if vis_known_mask.sum() > 0 else 0.0
    prec = prec if vis_known_mask.sum() > 0 else 0.0
    rec  = rec  if vis_known_mask.sum() > 0 else 0.0
    f1   = f1   if vis_known_mask.sum() > 0 else 0.0
    far  = far  if vis_known_mask.sum() > 0 else 0.0

    # Confidence histogram: separate hidden vs visible unknowns
    all_conf = np.max(all_probs, axis=-1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(all_conf[known_mask & is_visible_sample],  bins=60, alpha=0.6,
            label="Visible - classified", color="steelblue", edgecolor="white")
    ax.hist(all_conf[unknown_mask & is_visible_sample], bins=60, alpha=0.6,
            label="Visible - flagged unknown", color="orange", edgecolor="white")
    if is_hidden_sample.sum() > 0:
        ax.hist(all_conf[is_hidden_sample], bins=60, alpha=0.7,
                label=f"Hidden classes {hidden_names}", color="salmon",
                edgecolor="white")
    ax.axvline(conf_det.threshold, color="red", lw=2, linestyle="--", label=f"tau={conf_det.threshold:.3f}")
    ax.set_title("Policy Confidence Distribution - Test Set")
    ax.set_xlabel("Confidence")
    ax.legend(fontsize=8)
    ent_path = os.path.join(LOG_DIR, "eval_confidence_distribution.png")
    fig.savefig(ent_path, dpi=120)
    plt.close(fig)
    print(f"  [OK] Confidence plot -> {ent_path}")

    # ===================================================================
    # PHASE 2 - Open-set discovery on unknown latent vectors
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2 - Continual class discovery (open-set)")
    print(f"{'='*60}")

    unknown_z    = all_z[unknown_mask]
    unknown_true = y_seq[unknown_mask]     # original label indices

    print(f"\n  Unknown samples collected : {len(unknown_z):,}")
    if hidden_total > 0:
        hidden_in_unknown = is_hidden_sample[unknown_mask].sum()
        print(f"  Of which truly hidden     : {hidden_in_unknown:,} / {hidden_total:,} "
              f"({100*hidden_in_unknown/max(hidden_total,1):.1f}%)")

    unk_buf = UnknownBuffer(trigger_size=len(unknown_z), max_size=len(unknown_z) + 1)
    unk_buf.add_batch(unknown_z, unknown_true)

    discovery = ContinualClassDiscovery(
        sac_agent         = sac,
        env               = env,
        dbscan_eps        = dbscan_eps,
        dbscan_min_samples= dbscan_min_samples,
        min_cluster_size  = min_cluster_size,
        max_new_per_event = 10,
    )

    Z_unk, y_unk, _ = unk_buf.flush()
    n_new = discovery.discover(Z_unk, y_unk)

    print(f"\n  New clusters discovered : {n_new}")
    print(f"  Total action space now  : {sac.num_actions}  "
          f"({num_known} trained + {n_new} discovered)")
    if hidden_names:
        print(f"\n  -- Hidden class recovery check --")
        print(f"  Hidden classes were : {hidden_names}")
        print(f"  Clusters found      : {n_new}")
        if n_new >= len(hidden_names):
            print(f"  [GOOD] Enough clusters found to potentially represent "
                  f"all {len(hidden_names)} hidden class(es)")
        else:
            print(f"  [PARTIAL] Only {n_new}/{len(hidden_names)} hidden classes "
                  f"may have been discovered (need more training epochs)")

    # Unknown Attack Detection Rate:
    # What fraction of hidden-class test samples were correctly flagged as unknown?
    if hidden_total > 0:
        unk_detection_rate = hidden_flagged / hidden_total
        print(f"\n  Unknown Attack Detection Rate")
        print(f"  (hidden samples flagged): {unk_detection_rate*100:.2f}%")
    else:
        unk_detection_rate = unknown_mask.mean()
        print(f"\n  Unknown Detection Rate : {unk_detection_rate*100:.2f}%")

    # Cluster centroid visualisation
    c2d = discovery.get_centroids_2d()
    if c2d is not None and len(c2d) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(c2d[:, 0], c2d[:, 1], c=range(len(c2d)),
                        cmap="tab10", s=100, zorder=2, edgecolors="black")
        for i, (x_, y_) in enumerate(c2d):
            ax.annotate(f"C{i}", (x_, y_), fontsize=9,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"Discovered Cluster Centroids (PCA-2D)  n={len(c2d)}")
        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")
        plt.tight_layout()
        cl_path = os.path.join(LOG_DIR, "cluster_visualization.png")
        fig.savefig(cl_path, dpi=120)
        plt.close(fig)
        print(f"  [OK] Cluster visualization -> {cl_path}")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete.")
    print(f"{'='*60}\n")

    return {
        "accuracy":  acc  if known_mask.sum() > 0 else 0.0,
        "precision": prec if vis_known_mask.sum() > 0 else 0.0,
        "recall":    rec  if vis_known_mask.sum() > 0 else 0.0,
        "f1":        f1   if vis_known_mask.sum() > 0 else 0.0,
        "far":       far  if vis_known_mask.sum() > 0 else 0.0,
        "n_unknown_flagged":      int(unknown_mask.sum()),
        "n_new_classes":          n_new,
        "unknown_detection_rate": float(unk_detection_rate),
    }


# =======================================================================
# CLI
# =======================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CORL-IDS Evaluation + Discovery")
    p.add_argument("--model",             type=str,   default=os.path.join(MODEL_DIR, "trained_model.pt"))
    p.add_argument("--device",            type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dbscan-eps",        type=float, default=1.2)
    p.add_argument("--dbscan-min-samples",type=int,   default=30)
    p.add_argument("--min-cluster-size",  type=int,   default=50)
    p.add_argument("--beta-entropy",      type=float, default=1.0)
    p.add_argument("--proto-margin",      type=float, default=2.0,
                   help="Prototype detector margin (higher=fewer unknowns flagged). Default: 2.0")
    args = p.parse_args()

    evaluate(
        ckpt_path         = args.model,
        device_str        = args.device,
        dbscan_eps        = args.dbscan_eps,
        dbscan_min_samples= args.dbscan_min_samples,
        min_cluster_size  = args.min_cluster_size,
        beta_entropy      = args.beta_entropy,
        proto_margin      = args.proto_margin,
    )
