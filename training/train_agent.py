# -*- coding: utf-8 -*-
"""
train.py  --  CORL-IDS Training Pipeline (Known Classes Only)

Usage:
    python src/train.py [--epochs N] [--batch-size B]

Design:
    Training is performed ONLY on the known labelled attack classes.
    The SAC agent learns to classify known traffic using rarity-aware rewards.
    Unknown-class discovery is deferred to the evaluation / inference phase.

Loop per epoch:
    1. Build sequences from training data
    2. Encode with LSTM  ->  z_t  (latent state)
    3. Actor selects action (known class)
    4. IDS environment returns rarity-scaled reward
    5. Store transition in replay buffer
    6. Update SAC networks every `update_every` steps
    7. Log + checkpoint

NO cluster discovery, NO unknown buffer during training.
"""

import os
import sys
import argparse
import math
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Local imports -------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.dataset_loader import load_dataset
from data.preprocessing import preprocess
from data.sequence_builder import build_sequences
from models.lstm_encoder import LSTMEncoder
from models.sac_agent import DiscreteSAC
from rewards.rarity_reward import RarityReward
from training.ids_environment import IDSEnvironment
from training.replay_buffer import ReplayBuffer


# =======================================================================
# Hyper-parameters
# =======================================================================

DEFAULTS = dict(
    epochs          = 100,
    batch_size      = 256,
    seq_len         = 10,
    latent_dim      = 32,
    lstm_hidden     = 64,
    lr              = 3e-4,
    gamma           = 0.99,
    alpha_entropy   = 0.2,
    alpha_min       = 0.005,       # Allow policy to become more confident
    lambda_rarity   = 2.0,         # Max scaling to force minority attention
    lambda_contrast = 2.0,         # Push embeddings much further apart to isolate unknowns
    buffer_capacity = 100_000,
    update_every    = 4,
    seed            = 42,
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    log_interval    = 10,
    save_interval   = 10,
    # Classes to hide from training (kept in test set as unknowns)
    hidden_classes  = ["ransomware", "ddos"],
)


# =======================================================================
# Helpers
# =======================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(arr, device, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=device)


# -----------------------------------------------------------------------
# FIX 2: Supervised Contrastive Loss on LSTM encoder
# -----------------------------------------------------------------------
def supervised_contrastive_loss(z, labels, temperature=0.1):
    """
    Pull same-class embeddings together, push different-class apart.
    Forces backdoor/password to have different latent representations
    even when their raw features are similar.
    z      : (B, latent_dim)  embeddings
    labels : (B,)             integer class labels
    """
    z = nn.functional.normalize(z, dim=1)
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z.device)
    sim      = torch.matmul(z, z.T) / temperature
    labels   = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)
    neg_mask = 1.0 - torch.eye(B, device=z.device)
    exp_sim  = torch.exp(sim) * neg_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    n_pos    = pos_mask.sum(dim=1)
    loss     = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
    return loss[n_pos > 0].mean() if (n_pos > 0).any() else torch.tensor(0.0, device=z.device)


def save_checkpoint(path, encoder, sac, scaler, le, cfg, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "sac":     sac.state_dict_full(),
        "cfg":     cfg,
        "meta":    meta,
    }, path)
    with open(path.replace(".pt", "_sklearn.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "le": le}, f)
    print(f"  [OK] Checkpoint saved -> {path}")


def save_plots(logs, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if logs["critic_loss"]:
        axes[0, 0].plot(logs["critic_loss"])
    axes[0, 0].set_title("Critic Loss")
    axes[0, 0].set_xlabel("Update step")

    if logs["actor_loss"]:
        axes[0, 1].plot(logs["actor_loss"])
    axes[0, 1].set_title("Actor Loss")
    axes[0, 1].set_xlabel("Update step")

    axes[1, 0].plot(logs["reward"])
    axes[1, 0].set_title("Mean Episode Reward per Epoch")
    axes[1, 0].set_xlabel("Epoch")

    if logs["alpha"]:
        axes[1, 1].plot(logs["alpha"])
    axes[1, 1].set_title("Entropy Temperature alpha")
    axes[1, 1].set_xlabel("Update step")

    plt.tight_layout()
    fig.savefig(os.path.join(log_dir, "training_curves.png"), dpi=120)
    plt.close(fig)
    print(f"  [OK] Training curves saved -> {log_dir}/training_curves.png")


# =======================================================================
# Main training function
# =======================================================================

def train(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    print(f"\n{'='*60}")
    print(f"  CORL-IDS Training  (known classes only)")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    # -- Paths -----------------------------------------------------------
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path  = os.path.join(base_dir, "TON_IoT_Network_FULL.csv")
    model_dir = os.path.join(base_dir, "models")
    log_dir   = os.path.join(base_dir, "logs")
    ckpt_path = os.path.join(model_dir, "trained_model.pt")

    # -- 1. Load data ----------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test, le, class_probs, feat_cols, hidden_ids = \
        load_dataset(csv_path,
                     test_size=0.2,
                     seed=cfg["seed"],
                     hidden_classes=cfg.get("hidden_classes", []))

    # -- 2. Preprocess ---------------------------------------------------
    X_train_np, X_test_np, scaler, feature_dim = preprocess(X_train_raw, X_test_raw)

    # num_classes = number of VISIBLE classes the SAC will be trained on
    hidden_names  = cfg.get("hidden_classes", [])
    visible_names = [c for c in le.classes_ if c not in hidden_names]
    num_classes   = len(visible_names)
    # Build a label remapping: original idx -> visible idx
    # (hidden class indices are simply not seen during training)
    visible_ids   = sorted([int(le.transform([c])[0]) for c in visible_names])
    orig_to_vis   = {orig: vis for vis, orig in enumerate(visible_ids)}

    # Remap y_train labels to contiguous visible indices
    y_train_vis = np.array([orig_to_vis[lbl] for lbl in y_train], dtype=np.int64)

    print(f"Visible classes  : {num_classes} {visible_names}")
    print(f"Hidden classes   : {len(hidden_ids)} {hidden_names}")
    print(f"Feature dim      : {feature_dim}")

    # -- 3. Build sequences ----------------------------------------------
    print("Building sequences ...")
    X_seq_train, y_seq_train = build_sequences(X_train_np, y_train_vis, cfg["seq_len"])
    print(f"Train seqs: {len(y_seq_train):,}")

    # -- 4. Build modules ------------------------------------------------
    encoder  = LSTMEncoder(feature_dim, cfg["lstm_hidden"], cfg["latent_dim"]).to(device)
    # Upgraded to 2-layer MLP for higher projection capacity
    encoder_classifier = torch.nn.Sequential(
        torch.nn.Linear(cfg["latent_dim"], cfg["latent_dim"] // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg["latent_dim"] // 2, num_classes)
    ).to(device)

    # Map original index probabilities to visible index probabilities purely
    # based on the original un-sampled training distribution
    class_probs_vis = {orig_to_vis[orig]: p for orig, p in class_probs.items() if orig in orig_to_vis}
    
    # Pre-compute inverse probability weights for balanced replay sampling
    # (Removed to avoid double compensation; CORL-IDS uses RarityReward directly)
    class_weights = None
    
    ce_weights = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for c, prob in class_probs_vis.items():
        ce_weights[c] = 1.0 / (max(prob, 1e-8) ** 0.5)
    ce_weights /= ce_weights.sum()

    encoder_criterion = torch.nn.CrossEntropyLoss(weight=ce_weights)
    enc_optim = torch.optim.Adam(list(encoder.parameters()) + list(encoder_classifier.parameters()), lr=cfg["lr"])

    rarity  = RarityReward(class_probs_vis, lambda_=cfg["lambda_rarity"])
    env     = IDSEnvironment(num_classes, rarity_reward=rarity)

    sac = DiscreteSAC(
        state_dim   = cfg["latent_dim"],
        num_actions = num_classes,
        lr          = cfg["lr"],
        gamma       = cfg["gamma"],
        alpha       = cfg["alpha_entropy"],
        auto_alpha  = True,
        device      = cfg["device"],
    )
    alpha_min = cfg.get("alpha_min", 0.01)  # FIX 3: floor

    replay = ReplayBuffer(cfg["buffer_capacity"], cfg["latent_dim"], device=cfg["device"])

    # -- 5. Training loop ------------------------------------------------
    logs = {"critic_loss": [], "actor_loss": [], "reward": [], "alpha": []}
    N_train   = len(y_seq_train)
    total_steps = 0
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        perm       = np.random.permutation(N_train)
        X_shuffled = X_seq_train[perm]
        y_shuffled = y_seq_train[perm]

        epoch_rewards = []
        num_batches   = math.ceil(N_train / cfg["batch_size"])

        for batch_idx in range(num_batches):
            s  = batch_idx * cfg["batch_size"]
            e  = min(s + cfg["batch_size"], N_train)
            xb = torch.tensor(X_shuffled[s:e], dtype=torch.float32, device=device)
            yb = y_shuffled[s:e]

            # a) Encode + CrossEntropy representation clustering
            xb_t = torch.tensor(X_shuffled[s:e], dtype=torch.float32, device=device)
            yb_t = torch.tensor(y_shuffled[s:e], dtype=torch.long,    device=device)
            yb   = y_shuffled[s:e]

            encoder.train()
            z_batch_t = encoder(xb_t)
            logits    = encoder_classifier(z_batch_t)
            c_loss    = encoder_criterion(logits, yb_t)
            enc_optim.zero_grad()
            (cfg.get("lambda_contrast", 0.5) * c_loss).backward()
            enc_optim.step()

            # b) Vectorized Step through environment directly on GPU
            z_batch_t_detached = z_batch_t.detach()
            encoder.eval()

            states_t = env.reset_batch(z_batch_t_detached, yb_t)
            actions_t, _probs_t = sac.select_action_batch(states_t, deterministic=False)
            next_states_t, rewards_t, dones_t, _info = env.step_batch(actions_t)

            # c) Push to vectorized replay buffer directly from GPU
            replay.push_batch(
                states_t,
                actions_t,
                rewards_t,
                next_states_t,
                dones_t,
                true_labels=yb_t,
            )

            # d) SAC update + continuous exploration guarantee
            if len(replay) >= cfg["batch_size"] and (total_steps % cfg["update_every"] == 0):
                states_s, acts_s, rews_s, next_s, dones_s = replay.sample(cfg["batch_size"], class_weights=ce_weights)
                info = sac.update(
                    states_s,
                    acts_s,
                    rews_s,
                    next_s,
                    dones_s,
                )
                # Clamp log_alpha so policy keeps some exploration
                if hasattr(sac, "log_alpha"):
                    with torch.no_grad():
                        sac.log_alpha.clamp_(min=math.log(alpha_min))
                logs["critic_loss"].append(info["critic_loss"])
                logs["actor_loss"].append(info["actor_loss"])
                logs["alpha"].append(info["alpha"])

            epoch_rewards.extend(rewards_t.cpu().tolist())
            total_steps += 1

        # -- Epoch summary -----------------------------------------------
        mean_r = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        logs["reward"].append(mean_r)

        if epoch % cfg["log_interval"] == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"reward={mean_r:+.4f} | "
                  f"alpha={sac.alpha:.4f} | "
                  f"time={elapsed:.1f}s")

        # -- Checkpoint --------------------------------------------------
        if epoch % cfg["save_interval"] == 0 or epoch == cfg["epochs"]:
            meta = {
                "epoch":          epoch,
                "num_classes":    num_classes,
                "feature_dim":    feature_dim,
                "le_classes":     list(le.classes_),
                "visible_names":  visible_names,
                "visible_ids":    visible_ids,
                "hidden_names":   hidden_names,
                "hidden_ids":     hidden_ids,
            }
            save_checkpoint(ckpt_path, encoder, sac, scaler, le, cfg, meta)

    # -- Final plots -----------------------------------------------------
    save_plots(logs, log_dir)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Visible classes trained : {num_classes}  {visible_names}")
    print(f"  Hidden classes          : {hidden_names} (will appear as UNKNOWN in eval)")
    print(f"  Model saved to          : {ckpt_path}")
    print(f"  Run 'python src/evaluate.py' to evaluate + discover unknown attacks.")
    print(f"{'='*60}\n")


# =======================================================================
# CLI
# =======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CORL-IDS Training (known classes)")
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--seq-len",        type=int,   default=DEFAULTS["seq_len"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--gamma",          type=float, default=DEFAULTS["gamma"])
    p.add_argument("--alpha",          type=float, default=DEFAULTS["alpha_entropy"])
    p.add_argument("--lambda-rarity",  type=float, default=DEFAULTS["lambda_rarity"])
    p.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    p.add_argument("--device",         type=str,   default=DEFAULTS["device"])
    p.add_argument("--hidden-classes", type=str,   nargs="*",
                   default=DEFAULTS["hidden_classes"],
                   help="Space-separated class names to hide from training. "
                        "Example: --hidden-classes ransomware ddos")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(DEFAULTS)
    cfg.update({
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "seq_len":        args.seq_len,
        "lr":             args.lr,
        "gamma":          args.gamma,
        "alpha_entropy":  args.alpha,
        "lambda_rarity":  args.lambda_rarity,
        "seed":           args.seed,
        "device":         args.device,
        "hidden_classes": args.hidden_classes or [],
    })
    train(cfg)
