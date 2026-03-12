# CORL-IDS: Continual Open-Set Reinforcement Learning Intrusion Detection System

A research-grade implementation combining:
- **LSTM temporal encoder** for network traffic representation
- **Discrete Soft Actor-Critic (SAC)** for RL-based classification
- **Entropy-based novelty detection** with adaptive thresholds
- **DBSCAN continual class discovery** with EWC-lite catastrophic forgetting prevention

---

## Project Structure

```
CORL-IDS/
├── train_test_network.csv       ← dataset (place here)
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py           ← CSV loading, label detection, train/test split
│   ├── preprocessing.py         ← Missing value handling, one-hot, StandardScaler
│   ├── sequence_builder.py      ← Sliding window (seq_len=10) sequence generation
│   ├── lstm_encoder.py          ← Module 1: LSTM → latent z_t ∈ R^32
│   ├── ids_environment.py       ← Module 2: Gym-like IDS environment
│   ├── rarity_reward.py         ← Module 3: Rarity-aware reward shaping
│   ├── sac_discrete.py          ← Module 4: Discrete SAC (twin critics + auto-α)
│   ├── replay_buffer.py         ← Module 5: Experience replay (capacity 100k)
│   ├── entropy_detector.py      ← Module 6: Adaptive entropy novelty detection
│   ├── unknown_buffer.py        ← Module 7: Buffer for unknown samples
│   ├── cluster_discovery.py     ← Module 8: DBSCAN class discovery + EWC-lite
│   ├── train.py                 ← Main training pipeline
│   └── evaluate.py              ← Evaluation & metrics
├── models/
│   └── trained_model.pt         ← saved after training
└── logs/
    ├── training_curves.png
    ├── entropy_distribution.png
    ├── cluster_visualization.png
    ├── confusion_matrix.png
    └── eval_entropy_distribution.png
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python src/train.py
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 64 | Mini-batch size |
| `--seq-len` | 10 | Sliding window length |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--alpha` | 0.2 | Initial entropy temperature |
| `--lambda-rarity` | 0.5 | Rarity reward coefficient λ |
| `--beta-entropy` | 1.0 | Novelty threshold multiplier β |
| `--device` | auto | `cuda` or `cpu` |

**Example (quick smoke test):**
```bash
python src/train.py --epochs 1 --batch-size 64
```

---

## Evaluation

```bash
python src/evaluate.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `models/trained_model.pt` | Path to checkpoint |
| `--device` | auto | `cuda` or `cpu` |

---

## Training Parameters Used

```python
batch_size         = 64
sequence_length    = 10
latent_dim         = 32        # LSTM projection dimension
lstm_hidden        = 64        # LSTM hidden size
learning_rate      = 3e-4
gamma              = 0.99
alpha_entropy      = 0.2       # SAC entropy temperature (auto-tuned)
lambda_rarity      = 0.5       # rarity reward scaling
beta_entropy       = 1.0       # adaptive threshold: τ = μ_H + β·σ_H
buffer_capacity    = 100_000
training_epochs    = 100
```

---

## Architecture Overview

```
Network Traffic CSV
        │
        ▼
[data_loader + preprocessing]
        │
        ▼  (seq_len=10, feature_dim=D)
[sequence_builder]
        │  X ∈ R^(N, 10, D)
        ▼
[LSTMEncoder]  →  z_t ∈ R^32
        │
        ├──→ [IDSEnvironment]  ←─ [RarityReward]
        │           │
        │     action, reward
        │           │
        ├──→ [EntropyDetector] → is_unknown?
        │           │                 │
        │     H > τ: YES       [UnknownBuffer]
        │                             │  (trigger)
        │                      [ContinualClassDiscovery]
        │                        DBSCAN → expand actions
        │
        └──→ [ReplayBuffer]
                    │
              [DiscreteSAC Update]
               Actor + Twin Critics
               + Temperature α
               + EWC-lite penalty
```

---

## Output Files

| File | Description |
|------|-------------|
| `models/trained_model.pt` | Full model checkpoint (encoder + SAC) |
| `models/trained_model_sklearn.pkl` | Scaler + LabelEncoder (sklearn objects) |
| `logs/training_curves.png` | Critic loss, actor loss, reward, α over training |
| `logs/entropy_distribution.png` | Policy entropy histogram (training) |
| `logs/cluster_visualization.png` | PCA-2D view of discovered cluster centroids |
| `logs/confusion_matrix.png` | Per-class confusion matrix |
| `logs/eval_entropy_distribution.png` | Policy entropy histogram (test set) |

---

## Metrics Reported

- **Accuracy** (known classes)
- **Precision, Recall, F1-Score** (macro averages)
- **False Alarm Rate** (FP / (FP + TN))
- **Unknown Attack Detection Rate** (fraction of test samples flagged as novel)
- Full per-class **Classification Report**

---

## Citation / Research Notes

This system implements the CORL-IDS architecture:
- **Continual learning**: DBSCAN discovers new attack classes at runtime; EWC-lite prevents catastrophic forgetting of old classes.
- **Open-set detection**: Entropy of the SAC policy distribution serves as a novelty score. High-entropy → unknown attack.
- **Rarity-aware RL**: Reward magnitude scales inversely with class frequency, improving detection of rare attacks.
