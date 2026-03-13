# CORL-IDS: Continual Open-Set Reinforcement Learning Intrusion Detection System

## 📌 Project Overview
CORL-IDS is a robust Intrusion Detection System built for extreme class imbalance and zero-day attack discovery. It leverages a Reinforcement Learning architecture backed by an LSTM encoder and a Discrete Soft Actor-Critic (SAC) agent. 

Unlike traditional methods that use synthetic data (like SMOTE) for class imbalance, CORL-IDS relies strictly on **Rarity-Aware Reward Shaping**, mathematically damping classification bias while preventing reward explosion. Additionally, it features a dual-phase evaluation methodology that detects unknown attacks via representation-space centroids and action-confidence bounds, feeding them into a density-based cluster discovery module to continually learn zero-day intrusions.

---

## 🚀 Installation & Dependencies

**Dependencies:**
- Python 3+
- `torch` >= 2.0.0
- `numpy` >= 1.24.0
- `pandas` >= 2.0.0
- `scikit-learn` >= 1.3.0
- `matplotlib` >= 3.7.0
- `seaborn` >= 0.12.0

**Installation:**
```bash
git clone https://github.com/khalil0401/CORL-IDS.git
cd CORL-IDS
pip install -r requirements.txt
```

---

## 📂 File Structure

```text
CORL-IDS/
├── train_test_network.csv          # Main dataset structure required for the pipeline
├── CORL-IDS_Kaggle.ipynb           # Kaggle Notebook wrapper for seamless execution
├── requirements.txt                # Python dependencies
├── data/
│   ├── dataset_loader.py           # Loads CSV, parses features, and handles class hiding
│   ├── preprocessing.py            # StandardScaler normalisation
│   └── sequence_builder.py         # Builds sliding LSTM sequences
├── detection/
│   ├── centroid_detector.py        # Mahalanobis distance bounds to training class representations
│   ├── cluster_discovery.py        # DBSCAN-based continual class expansion from unknowns
│   ├── confidence_unknown.py       # Softmax confidence bound threshold evaluator
│   ├── entropy_detector.py         # Legacy entropy-based anomaly detector
│   └── prototype_detector.py       # Legacy euclidean prototype anomaly detector
├── evaluation/
│   └── evaluate_ids.py             # Phase 1 Inference & Phase 2 Zero-Day Discovery pipeline
├── models/
│   ├── lstm_encoder.py             # Extractor yielding latent representations (z_t)
│   ├── sac_agent.py                # Discrete Soft Actor-Critic algorithm
│   ├── trained_model.pt            # Serialized trained PyTorch checkpoint
│   └── trained_model_sklearn.pkl   # Serialized StandardScaler and LabelEncoder
├── rewards/
│   └── rarity_reward.py            # Reward calculator using sqrt(p(y_t)) inverse scaling
├── training/
│   ├── ids_environment.py          # Simulated environment returning RL steps and rewards
│   ├── replay_buffer.py            # Vectorized PyTorch GPU transition buffer
│   ├── train_agent.py              # Main training script (Known classes only)
│   └── unknown_buffer.py           # Stores unknown-flagged embeddings for DBSCAN
└── logs/                           # Directory containing output evaluation plots
```

---

## 🧠 Modules Description

### `data/`
- **`dataset_loader.py`**: Reads `train_test_network.csv`, applies `LabelEncoder`, and dynamically extracts `hidden_classes` from the training set so they act as genuine zero-day attacks during testing.
- **`preprocessing.py`**: Applies numerical normalisation using `StandardScaler`.
- **`sequence_builder.py`**: Iterates through the raw time-series matrices to return sliding-window sequences necessary for the LSTM context extraction.

### `models/`
- **`lstm_encoder.py`**: Transforms sequences of network frames into dense latent representations $z_t$.
- **`sac_agent.py`**: Features the continuous-space Actor and Critic networks adapted for discrete categorical selections using reparameterized Softmax probabilities. Expanded dynamically during Phase 2 discovery.

### `rewards/`
- **`rarity_reward.py`**: Mathematical environment component determining the reinforcement signal. Calculates class probabilities $p(y_t)$ and applies bounded logarithmic scaling: $R_t = R_{base} \times (1 + \lambda \log(1 + 1 / \sqrt{p(y_t)}))$.

### `training/`
- **`train_agent.py`**: Core training loop. Iterates batches entirely in PyTorch tensors on GPU. Uses Supervised Contrastive Loss to separate distinct classes within the LSTM latent space prior to SAC steps.
- **`ids_environment.py`**: Encapsulates state-transitions. Resets on new batches and steps according to SAC network predictions, scoring actions against `rarity_reward.py`.
- **`replay_buffer.py`**: High-performance transition memory optimized for batch retrieval based on class probability weights.
- **`unknown_buffer.py`**: Caches samples that inference mechanisms determine are "Zero-Day/Unknown" for eventual cluster resolution.

### `detection/`
- **`confidence_unknown.py`**: Flags test samples as UNKNOWN if the maximum Soft Actor-Critic policy prediction confidence drops below a specified threshold (`tau_conf = 0.98`).
- **`centroid_detector.py`**: Maps the class center coordinates ($\mu$) in the latent space and utilizes Mahalanobis divergence ($\Sigma^{-1}$) to identify samples whose $z_t$ representation crosses acceptable $\sigma$ standard deviation boundaries (`distance_multiplier = 1.0`).
- **`cluster_discovery.py`**: Once unknown attacks are flagged, it pulls them from the `unknown_buffer` and applies DBSCAN density clustering (`dbscan_eps = 1.2`, `min_cluster_size = 100`). It then expands the action space natively for the `sac_agent` and `ids_environment`.
- **`entropy_detector.py` / `prototype_detector.py`**: Legacy detection schemas provided for architectural parity comparisons against prior versions.

### `evaluation/`
- **`evaluate_ids.py`**: Two-tiered evaluation pipeline.
    - **Phase 1**: Assesses trained known classifications via precision, recall, and False Alarm Ratings. Utilizes `confidence_unknown.py` and `centroid_detector.py` jointly to construct an `unknown_mask`.
    - **Phase 2 (Open-Set)**: Activates `cluster_discovery` on the filtered unknown pool to discover how many hidden distributions were contained inside the network traffic, establishing the final Unknown Attack Detection Rate (UADR).

---

## ⚙️ Running the Scripts & Usage

### 1. Training the Agent
Training is executed entirely on the visible (known) class segments. By default, `ransomware` and `ddos` are hidden to simulate zero-day circumstances.
```bash
python training/train_agent.py --epochs 100 --batch-size 256
```
**Optional Arguments:**
- `--epochs N`: Number of loops.
- `--batch-size B`: Training batch dimension.
- `--seq-len S`: Length of LSTM sequences (Default: 10).
- `--hidden-classes ransomware ddos`: Instructs the data loader to entirely eradicate these classifications from the SAC training memory.

*Outputs:*
- `models/trained_model.pt`
- `models/trained_model_sklearn.pkl`
- `logs/training_curves.png`

### 2. Evaluation & Anomaly Discovery
After generating the `trained_model.pt` checkpoint, use `evaluate_ids.py` to evaluate the system's resilience to zero-day data.
```bash
python evaluation/evaluate_ids.py
```
**Optional Arguments:**
- `--dbscan-eps`: Neighborhood radius for discovering isolated attack vector geometries (Default: `1.2`).
- `--min-cluster-size`: Minimum volume points required to instantiate a brand-new threat classification natively into the SAC (Default: `50`).

*Outputs:*
- Terminal: `Accuracy`, `F1-Score`, `False Alarm Rate`, `Unknown Attack Detection Rate (UADR)`, and the quantity of new zero-day clusters resolved.
- `logs/confusion_matrix.png`: Heatmap distribution of known class accuracy.
- `logs/eval_confidence_distribution.png`: Histogram distribution highlighting how unknown data was structurally filtered.
- `logs/cluster_visualization.png`: 2D PCA representation mapping the geometry of newly learned attack classes.

---

## 📊 Evaluation Procedures & Outputs
The inference scripts heavily detail mathematical metrics to confirm validity. Below is an example of an evaluation process capturing exactly 2 hidden classes natively across training thresholds:

```text
============================================================
  PHASE 1 - Known-class classification
============================================================

  Confidence threshold (tau_conf)  : 0.98
  Flagged by representation distance (Centroid) : 9,801
  Flagged by absolute policy confidence         : 2,619
  Flagged by EITHER (union)        : 10,547 / 42,200 (25.0%)
  Hidden-class samples : 7,999 in test set
  Hidden flagged UNKNOWN : 6,450 / 7,999 (80.6%)  <-- goal: HIGH

------------------------------------------------------------
  Accuracy  (visible, non-flagged) : 96.13%
  Macro F1-Score                   : 88.54%
  False Alarm Rate                 : 1.4216%

============================================================
  PHASE 2 - Continual class discovery (open-set)
============================================================

[DISCOVERY] Found 2 stable new cluster(s) from 10547 unknown samples.
[SAC] Action space expanded to 10
[ENV] Expanding action space: 8 -> 10

  -- Hidden class recovery check --
  Hidden classes were : ['ransomware', 'ddos']
  Clusters found      : 2
  [GOOD] Enough clusters found to potentially represent all 2 hidden class(es)

  Unknown Attack Detection Rate
  (hidden samples flagged): 80.64%
```

*(Visualizations for Training Curves, Confusion Matrix, and Open-Set Clusters are available dynamically in the `logs/` directory post-completion).*
