# -*- coding: utf-8 -*-
"""
run_continual_learning.py

CORL-IDS: Continual Open-Set Reinforcement Learning Master Pipeline Loop
Strict Execution & Phase Reporting

Phase 1: Base Training & Evaluation
Phase 2: Zero-Day Discovery & Unknown Detection Rates
Phase 3: Continual EWC Fine-Tuning & New Class Evaluation
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# -------------------------------------------------------------------
# STRICT REQUIREMENT: Verify all files and folders
# -------------------------------------------------------------------
REQUIRED_FOLDERS = ["models", "logs", "data", "training", "detection", "rewards"]
for fld in REQUIRED_FOLDERS:
    path = os.path.join(BASE_DIR, fld)
    if not os.path.isdir(path):
        print(f"[FATAL ERROR] Required folder missing: {path}")
        sys.exit(1)

# We are the run_continual_learning.py file, evaluate_ids.py is in evaluation/
if not os.path.isfile(os.path.join(BASE_DIR, "evaluation", "evaluate_ids.py")):
    print(f"[FATAL ERROR] evaluate_ids.py missing in evaluation/ folder.")
    sys.exit(1)

# Import necessary core modules inside a try-catch to strictly verify their presence
try:
    from data.dataset_loader import load_dataset
    from data.preprocessing import preprocess
    from data.sequence_builder import build_sequences
    from models.lstm_encoder import LSTMEncoder
    from models.sac_agent import DiscreteSAC
    from rewards.rarity_reward import RarityReward
    from training.ids_environment import IDSEnvironment
    from training.replay_buffer import ReplayBuffer
    from training.unknown_buffer import UnknownBuffer
    from detection.confidence_unknown import ConfidenceUnknownDetector
    from detection.centroid_detector import CentroidDetector
    from detection.cluster_discovery import ContinualClassDiscovery
    from training.train_agent import supervised_contrastive_loss, save_checkpoint
except ImportError as e:
    print(f"[FATAL ERROR] Missing required project module: {e}")
    sys.exit(1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_confusion_matrix(y_true, y_pred, labels, target_names, filepath, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)

def run_pipeline(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    print(f"\n{'='*70}\n  CORL-IDS Automated Continual Learning Pipeline\n  Device: {device}\n{'='*70}\n")

    csv_path  = os.path.join(BASE_DIR, "train_test_network.csv")
    model_dir = os.path.join(BASE_DIR, "models")
    log_dir   = os.path.join(BASE_DIR, "logs")

    print("[INIT] Loading Data Stream...")
    try:
        X_train_raw, X_test_raw, y_train, y_test, le, class_probs, feat_cols, hidden_ids = load_dataset(
            csv_path, test_size=0.2, seed=cfg["seed"], hidden_classes=cfg["hidden_classes"]
        )
        X_train_np, X_test_np, scaler, feature_dim = preprocess(X_train_raw, X_test_raw)
    except Exception as e:
        print(f"[FATAL ERROR] Dataset loading/preprocessing failed. {e}")
        sys.exit(1)

    visible_names = [c for c in le.classes_ if c not in cfg["hidden_classes"]]
    num_known_classes = len(visible_names)
    visible_ids = sorted([int(le.transform([c])[0]) for c in visible_names])
    orig_to_vis = {orig: vis for vis, orig in enumerate(visible_ids)}
    y_train_vis = np.array([orig_to_vis[lbl] for lbl in y_train], dtype=np.int64)

    print("[INIT] Building sequences...")
    try:
        X_seq_train, y_seq_train = build_sequences(X_train_np, y_train_vis, cfg["seq_len"])
        X_seq_test,  y_seq_test  = build_sequences(X_test_np, y_test, cfg["seq_len"])
    except Exception as e:
        print(f"[FATAL ERROR] Sequence building failed. {e}")
        sys.exit(1)

    # Dictionary to hold the final summary report metrics
    summary_metrics = {}

    print("[INIT] Building Core Components...")
    try:
        encoder = LSTMEncoder(feature_dim, cfg["lstm_hidden"], cfg["latent_dim"], use_attention=cfg["use_attention"]).to(device)
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"])

        class_probs_vis = {orig_to_vis[orig]: p for orig, p in class_probs.items() if orig in orig_to_vis}
        class_weights = torch.zeros(num_known_classes, dtype=torch.float32, device=device)
        for c, prob in class_probs_vis.items(): class_weights[c] = 1.0 / max(prob, 1e-8)
        class_weights /= class_weights.sum()

        rarity = RarityReward(class_probs_vis, lambda_=cfg["lambda_rarity"])
        env    = IDSEnvironment(num_known_classes, rarity_reward=rarity)
        sac    = DiscreteSAC(cfg["latent_dim"], num_known_classes, lr=cfg["lr"], gamma=cfg["gamma"], alpha=cfg["alpha_entropy"], auto_alpha=True, device=cfg["device"])
        replay = ReplayBuffer(cfg["buffer_capacity"], cfg["latent_dim"], device=cfg["device"])
        
        # Test Unknown Buffer initialization
        _test_unk = UnknownBuffer(trigger_size=1, max_size=2)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to initialize Core Components. {e}")
        sys.exit(1)


    # ===================================================================
    # PHASE 1: Base SAC Training & Evaluation
    # ===================================================================
    print(f"\n{'='*70}\n  PHASE 1 - Base Training (Known Classes)\n{'='*70}")
    N_train = len(y_seq_train)
    total_steps = 0
    alpha_min = cfg.get("alpha_min", 0.005)

    try:
        for epoch in range(1, cfg["epochs"] + 1):
            perm = np.random.permutation(N_train)
            X_shuffled, y_shuffled = X_seq_train[perm], y_seq_train[perm]
            epoch_rewards = []
            num_batches = math.ceil(N_train / cfg["batch_size"])

            for b in range(num_batches):
                s, e = b * cfg["batch_size"], min((b + 1) * cfg["batch_size"], N_train)
                xb_t = torch.tensor(X_shuffled[s:e], dtype=torch.float32, device=device)
                yb_t = torch.tensor(y_shuffled[s:e], dtype=torch.long, device=device)

                encoder.train()
                z_batch_t = encoder(xb_t)
                c_loss = supervised_contrastive_loss(z_batch_t, yb_t)
                enc_optim.zero_grad()
                (cfg["lambda_contrast"] * c_loss).backward()
                enc_optim.step()

                encoder.eval()
                states_t = env.reset_batch(z_batch_t.detach(), yb_t)
                actions_t, _ = sac.select_action_batch(states_t, deterministic=False)
                next_states_t, rewards_t, dones_t, _ = env.step_batch(actions_t)

                replay.push_batch(states_t, actions_t, rewards_t, next_states_t, dones_t, true_labels=yb_t)

                if len(replay) >= cfg["batch_size"] and (total_steps % cfg["update_every"] == 0):
                    sr, ar, rr, nr, dr = replay.sample(cfg["batch_size"], class_weights=class_weights)
                    sac.update(sr, ar, rr, nr, dr, ewc_loss=None)
                    if hasattr(sac, "log_alpha"):
                        with torch.no_grad(): sac.log_alpha.clamp_(min=math.log(alpha_min))
                    
                epoch_rewards.extend(rewards_t.cpu().tolist())
                total_steps += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{cfg['epochs']} | reward={np.mean(epoch_rewards):+.4f} | c_loss={c_loss.item():.4f}")

        phase1_ckpt = os.path.join(model_dir, "trained_model_phase1.pt")
        save_checkpoint(phase1_ckpt, encoder, sac, scaler, le, cfg, {"epoch": cfg["epochs"], "num_classes": num_known_classes, "feature_dim": feature_dim})
        print(f"  -> Model saved: {phase1_ckpt}")
    except Exception as e:
        print(f"[FATAL ERROR] Phase 1 Training failed. {e}")
        sys.exit(1)

    # Phase 1 Evaluation on Test Data (Known Classes Only)
    print("\n  -> Evaluating Phase 1 on Test Set...")
    try:
        encoder.eval()
        sac.actor.eval()
        all_preds, all_probs, all_z = [], [], []
        with torch.no_grad():
            for b in range(math.ceil(len(X_seq_test) / 512)):
                s, e = b * 512, min((b + 1) * 512, len(X_seq_test))
                xb_t = torch.tensor(X_seq_test[s:e], dtype=torch.float32, device=device)
                z_t = encoder(xb_t)
                acts_t, probs_t = sac.select_action_batch(z_t, deterministic=True)
                all_preds.extend(acts_t.cpu().tolist())
                all_probs.append(probs_t.cpu().numpy())
                all_z.append(z_t.cpu().numpy())
                
        all_z = np.concatenate(all_z, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.array(all_preds)

        is_visible_sample = np.isin(y_seq_test, visible_ids)
        y_test_vis_mask = y_seq_test[is_visible_sample]
        y_test_mapped = np.array([orig_to_vis.get(lbl, -1) for lbl in y_test_vis_mask])
        y_pred_mapped = all_preds[is_visible_sample]

        acc_p1 = accuracy_score(y_test_mapped, y_pred_mapped)
        prec_p1, rec_p1, f1_p1, _ = precision_recall_fscore_support(y_test_mapped, y_pred_mapped, average="macro", zero_division=0)
        
        # FAR calculation using normal
        normal_idx = None
        if "normal" in visible_names: normal_idx = orig_to_vis[int(le.transform(["normal"])[0])]
        elif "benign" in visible_names: normal_idx = orig_to_vis[int(le.transform(["benign"])[0])]
        
        if normal_idx is not None:
            FP = (((y_test_mapped != normal_idx) == False) & ((y_pred_mapped != normal_idx) == True)).sum()
            TN = (((y_test_mapped != normal_idx) == False) & ((y_pred_mapped != normal_idx) == False)).sum()
            far_p1 = FP / max(FP + TN, 1e-8)
        else:
            far_p1 = float('nan')

        print(f"     Accuracy : {acc_p1*100:.2f}%")
        print(f"     Precision: {prec_p1*100:.2f}%")
        print(f"     Recall   : {rec_p1*100:.2f}%")
        print(f"     F1-Score : {f1_p1*100:.2f}%")
        print(f"     FAR      : {far_p1*100:.4f}%" if not math.isnan(far_p1) else "     FAR      : N/A")

        summary_metrics["p1_acc"] = acc_p1
        summary_metrics["p1_f1"] = f1_p1
        summary_metrics["p1_far"] = far_p1

        # Save Phase 1 Confusion Matrix
        labels_in = list(range(num_known_classes))
        save_confusion_matrix(y_test_mapped, y_pred_mapped, labels_in, visible_names, os.path.join(log_dir, "confusion_phase1.png"), "Phase 1 - Known Classes")
        print(f"  -> Saved logs/confusion_phase1.png")

        # Save Phase 1 CSV
        report_dict = precision_recall_fscore_support(y_test_mapped, y_pred_mapped, labels=labels_in, zero_division=0)
        df_p1 = pd.DataFrame({
            "Class": visible_names,
            "Precision": report_dict[0],
            "Recall": report_dict[1],
            "F1-Score": report_dict[2],
            "Support": report_dict[3]
        })
        df_p1.to_csv(os.path.join(log_dir, "per_class_metrics_phase1.csv"), index=False)
        print(f"  -> Saved logs/per_class_metrics_phase1.csv")

    except Exception as e:
        print(f"[FATAL ERROR] Phase 1 Evaluation failed. {e}")
        sys.exit(1)


    # ===================================================================
    # PHASE 2: Open-Set Discovery (Zero-Day Detection)
    # ===================================================================
    print(f"\n{'='*70}\n  PHASE 2 - Open-Set Discovery (Zero-Day Detection)\n{'='*70}")
    try:
        conf_det = ConfidenceUnknownDetector(beta=cfg["beta_entropy"])
        centroid_det = CentroidDetector(distance_multiplier=1.0)
        
        # Fit Adapters natively mapped against knowns
        encoder.eval()
        Z_knowns, c_probs = [], []
        with torch.no_grad():
            for b in range(math.ceil(N_train / 512)):
                s, e = b * 512, min((b + 1) * 512, N_train)
                xb_t = torch.tensor(X_seq_train[s:e], dtype=torch.float32, device=device)
                z_k = encoder(xb_t)
                _, pb = sac.select_action_batch(z_k, deterministic=True)
                Z_knowns.append(z_k.cpu().numpy())
                c_probs.append(pb.cpu().numpy())
        conf_det.fit(np.concatenate(c_probs, axis=0))
        centroid_det.fit(np.concatenate(Z_knowns, axis=0), y_seq_train, num_classes=num_known_classes)

        conf_unknown     = conf_det.predict_batch(all_probs)
        centroid_unknown = centroid_det.predict_batch(all_z)
        unknown_mask     = conf_unknown | centroid_unknown

        num_unknown_flagged = unknown_mask.sum()
        is_hidden_sample = np.isin(y_seq_test, hidden_ids)
        total_hidden = is_hidden_sample.sum()
        hidden_detected = (unknown_mask & is_hidden_sample).sum()
        
        uadr = hidden_detected / max(total_hidden, 1)

        print(f"  -> Flagged Unknown Samples: {num_unknown_flagged}")
        print(f"  -> Hidden Classes Present : {total_hidden}")
        print(f"  -> Hidden Flagged         : {hidden_detected}")
        print(f"  -> UADR                   : {uadr*100:.2f}%")

        summary_metrics["p2_unk"] = num_unknown_flagged
        summary_metrics["p2_hid"] = hidden_detected
        summary_metrics["p2_uadr"] = uadr

        # Save Confidence Histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        test_conf = np.max(all_probs, axis=-1)
        ax.hist(test_conf[~unknown_mask], bins=50, alpha=0.6, label="Classified as Known", color="steelblue")
        ax.hist(test_conf[unknown_mask], bins=50, alpha=0.6, label="Flagged as Unknown", color="orange")
        ax.axvline(conf_det.threshold, color="red", linestyle="--", label=f"Threshold {conf_det.threshold:.2f}")
        ax.legend()
        ax.set_title("Phase 2 - Policy Confidence Distribution")
        fig.savefig(os.path.join(log_dir, "confidence_hist_phase2.png"), dpi=120)
        plt.close(fig)
        print(f"  -> Saved logs/confidence_hist_phase2.png")

        # Phase 2 Model Checkpoint
        phase2_ckpt = os.path.join(model_dir, "trained_model_phase2.pt")
        save_checkpoint(phase2_ckpt, encoder, sac, scaler, le, cfg, {"epoch": cfg["epochs"], "num_classes": num_known_classes, "feature_dim": feature_dim})
        print(f"  -> Model saved: {phase2_ckpt}")

    except Exception as e:
        print(f"[FATAL ERROR] Phase 2 Execution failed. {e}")
        sys.exit(1)


    # ===================================================================
    # PHASE 3: Continual Learning (EWC + New Classes)
    # ===================================================================
    print(f"\n{'='*70}\n  PHASE 3 - Continual Fine-Tuning\n{'='*70}")
    try:
        flagged_z = all_z[unknown_mask]
        flagged_idx = np.where(unknown_mask)[0]
        
        discovery = ContinualClassDiscovery(sac_agent=sac, env=env, rarity_reward=rarity, min_cluster_size=50, dbscan_eps=1.2, ewc_lambda=cfg["ewc_lambda"])
        n_new = discovery.discover(flagged_z, labels=np.full(len(flagged_z), -1))
        
        summary_metrics["p2_clusters"] = n_new
        print(f"  -> Initialized Continual Class Discovery. Stable Clusters: {n_new}")

        if n_new > 0:
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=1.2, min_samples=30, n_jobs=-1).fit(flagged_z)
            pseudo_labels = db.labels_

            # Save Clusters Image
            c2d = discovery.get_centroids_2d()
            if c2d is not None and len(c2d) >= 1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(c2d[:, 0], c2d[:, 1], c=range(len(c2d)), cmap="tab10", s=100, edgecolors="black")
                ax.set_title(f"Phase 2 - Discovered Clusters (n={len(c2d)})")
                fig.savefig(os.path.join(log_dir, "clusters_phase2.png"), dpi=120)
                plt.close(fig)
                print(f"  -> Saved logs/clusters_phase2.png")

            injected_count = 0
            for unique_c in [c for c in np.unique(pseudo_labels) if c != -1]:
                mask = pseudo_labels == unique_c
                if mask.sum() >= 50:
                    sac_class_id = num_known_classes + injected_count
                    centroid_det.incremental_update(sac_class_id, flagged_z[mask])
                    
                    xb_new = torch.tensor(X_seq_test[flagged_idx][mask], dtype=torch.float32, device=device)
                    encoder.eval()
                    with torch.no_grad(): z_new = encoder(xb_new)
                    acts = torch.full((len(z_new),), sac_class_id, dtype=torch.int64, device=device)
                    rews = torch.full((len(z_new),), 1.0, dtype=torch.float32, device=device)
                    replay.push_batch(z_new, acts, rews, z_new, torch.ones_like(rews), true_labels=acts)
                    injected_count += 1
                    
            print(f"  -> Expanded ReplayBuffer with {injected_count} new isolated attack geometries.")
            new_total_classes = sac.num_actions

            class_weights = torch.ones(new_total_classes, dtype=torch.float32, device=device)
            class_weights[num_known_classes:] = 5.0
            class_weights /= class_weights.sum()

            print(f"  -> Fine-Tuning online for {cfg['continual_epochs']} epochs with EWC parameters locked...")
            for epoch in range(1, cfg["continual_epochs"] + 1):
                cl_losses = []
                for _ in range(200):
                    sr, ar, rr, nr, dr = replay.sample(cfg["batch_size"], class_weights=class_weights)
                    ewc_val = discovery.ewc_penalty()
                    info = sac.update(sr, ar, rr, nr, dr, ewc_loss=ewc_val)
                    cl_losses.append(info["actor_loss"])

                if epoch % 5 == 0 or epoch == 1:
                    print(f"    Online Epoch {epoch:2d}/{cfg['continual_epochs']} | actor_loss={np.mean(cl_losses):.4f} | EWC_penalty={ewc_val.item():.4f}")

            # Phase 3 Eval
            print("\n  -> Re-Evaluating over Test Set to verify old knowledge preservation + new attack identification...")
            encoder.eval()
            sac.actor.eval()
            p3_preds = []
            with torch.no_grad():
                for b in range(math.ceil(len(X_seq_test) / 512)):
                    s, e = b * 512, min((b + 1) * 512, len(X_seq_test))
                    xb_t = torch.tensor(X_seq_test[s:e], dtype=torch.float32, device=device)
                    acts_t, _ = sac.select_action_batch(encoder(xb_t), deterministic=True)
                    p3_preds.extend(acts_t.cpu().tolist())
            p3_preds = np.array(p3_preds)

            p3_old_acc = accuracy_score(y_test_mapped, p3_preds[is_visible_sample])
            
            # Since pseudo-labels are arbitrary, we compute accuracy on new classes as 
            # the percentage of hidden samples that were mapped to ANY of the newly assigned > num_known boundaries
            p3_new_preds = p3_preds[is_hidden_sample]
            p3_new_acc = (p3_new_preds >= num_known_classes).mean() if len(p3_new_preds) > 0 else 0.0

            print(f"     Old Classes Accuracy : {p3_old_acc*100:.2f}%")
            print(f"     New Classes Accuracy : {p3_new_acc*100:.2f}%")

            summary_metrics["p3_old_acc"] = p3_old_acc
            summary_metrics["p3_new_acc"] = p3_new_acc
            summary_metrics["p3_udr"] = p3_new_acc # Identification as new classes is effectively the new UDR

            # Construct Confusion Matrices for Phase 3 (Old vs Discovered bounds)
            test_labels_p3 = np.array([orig_to_vis.get(lbl, num_known_classes) for lbl in y_seq_test])
            p3_preds_clamped = np.clip(p3_preds, 0, num_known_classes)
            labels_p3 = list(range(num_known_classes + 1))
            names_p3 = visible_names + ["Discovered/Zero-Day"]
            
            save_confusion_matrix(test_labels_p3, p3_preds_clamped, labels_p3, names_p3, os.path.join(log_dir, "confusion_phase3.png"), "Phase 3 - Old & Discovered Classes")
            print(f"  -> Saved logs/confusion_phase3.png")

            phase3_ckpt = os.path.join(model_dir, "trained_model_phase3.pt")
            save_checkpoint(phase3_ckpt, encoder, sac, scaler, le, cfg, {"epoch": cfg["epochs"]+cfg["continual_epochs"], "num_classes": new_total_classes, "feature_dim": feature_dim})
            print(f"  -> Model saved: {phase3_ckpt}")
        else:
            summary_metrics["p3_old_acc"] = summary_metrics["p1_acc"]
            summary_metrics["p3_new_acc"] = 0.0
            summary_metrics["p3_udr"] = 0.0
            print("  -> Bypass Continual EWC because no large clusters were verified.")

    except Exception as e:
        print(f"[FATAL ERROR] Phase 3 Continual Fine-Tuning failed. {e}")
        sys.exit(1)


    # ===================================================================
    # FINAL REPORTING
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"| {'Phase':<8} | {'Metric':<25} | {'Result':<15} |")
    print(f"|{'-'*10}|{'-'*27}|{'-'*17}|")
    print(f"| Phase1   | Accuracy                  | {summary_metrics['p1_acc']*100:5.2f}%         |")
    print(f"| Phase1   | F1-score                  | {summary_metrics['p1_f1']*100:5.2f}%         |")
    if math.isnan(summary_metrics['p1_far']):
        print(f"| Phase1   | FAR                       | N/A             |")
    else:
        print(f"| Phase1   | FAR                       | {summary_metrics['p1_far']*100:5.2f}%         |")
    print(f"| Phase2   | Unknown flagged           | {summary_metrics.get('p2_unk', 0):<15} |")
    print(f"| Phase2   | Hidden detected           | {summary_metrics.get('p2_hid', 0):<15} |")
    print(f"| Phase2   | UADR                      | {summary_metrics.get('p2_uadr', 0)*100:5.2f}%         |")
    print(f"| Phase2   | Clusters discovered       | {summary_metrics.get('p2_clusters', 0):<15} |")
    print(f"| Phase3   | Old class accuracy        | {summary_metrics.get('p3_old_acc', 0)*100:5.2f}%         |")
    print(f"| Phase3   | New class accuracy        | {summary_metrics.get('p3_new_acc', 0)*100:5.2f}%         |")
    print(f"| Phase3   | Unknown detection rate    | {summary_metrics.get('p3_udr', 0)*100:5.2f}%         |")
    print(f"{'='*70}")
    print("\nCORL-IDS Pipeline Completed Successfully")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--continual-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--lambda-rarity", type=float, default=2.0)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--ewc-lambda", type=float, default=0.4)
    p.add_argument("--beta-entropy", type=float, default=1.0)
    p.add_argument("--use-attention", action="store_true", default=True)
    p.add_argument("--hidden-classes", type=str, nargs="*", default=["ransomware", "ddos"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = {
        "epochs": args.epochs,
        "continual_epochs": args.continual_epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "gamma": args.gamma,
        "alpha_entropy": args.alpha,
        "lambda_rarity": args.lambda_rarity,
        "lambda_contrast": args.lambda_contrast,
        "ewc_lambda": args.ewc_lambda,
        "beta_entropy": args.beta_entropy,
        "use_attention": args.use_attention,
        "latent_dim": 32,
        "lstm_hidden": 64,
        "buffer_capacity": 100_000,
        "update_every": 4,
        "hidden_classes": args.hidden_classes,
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    run_pipeline(cfg)
