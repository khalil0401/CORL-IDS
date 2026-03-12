# CORL-IDS: Implementation vs. Paper Comparison

## Introduction
This document serves as a systematic comparison between the architecture and methodology described in the original CORL-IDS research paper and the current codebase implementation. 

The purpose of this review is to identify diverging methodologies, incomplete modules, or custom adaptations that differ from the literal text of the research paper. Reviewers should go through each component, verify the code against the paper's description, and update the **Implementation Check** (✅ / ❌ / ⚠️) and **Notes** sections accordingly.

---

## 1. Feature Encoding (LSTM)
**Description in Paper:**
The model utilizes a Long Short-Term Memory (LSTM) network to encode temporal sequences of network traffic into a dense latent representation vector ($z_t$). This representation captures sequential dependencies crucial for intrusion detection.

- **Implementation Check:** ✅
- **Notes:** Fully implemented. Uses the `LSTMEncoder` module with a default 64 hidden unit dimension and sliding-window `seq_len` parameters.

---

## 2. RL Agent (Soft Actor-Critic)
**Description in Paper:**
A discrete variant of the Soft Actor-Critic (SAC) reinforcement learning algorithm is employed. The agent balances exploration and exploitation by maximizing both the expected reward and the policy entropy. It outputs action probabilities (classification) based on the latent state $z_t$.

- **Implementation Check:** ✅
- **Notes:** Fully implemented. Uses `DiscreteSAC` with reparameterized Softmax probabilities acting on the discrete action space (classifying traffic types).

---

## 3. Rarity-Aware Reward Shaping
**Description in Paper:**
To handle extreme class imbalances naturally present in network traffic, the environment reward is scaled inversely proportional to the frequency (probability) of the true class. Rare attacks receive exponentially higher rewards to force the agent's attention without requiring synthetic data oversampling like SMOTE.

- **Implementation Check:** ✅
- **Notes:** Fully implemented inside `rewards/rarity_reward.py`. Uses the exact `sqrt(p)` inverse scaling formulation to prevent reward explosion for extreme minorities.

---

## 4. Critic Update (Twin Q-network)
**Description in Paper:**
To mitigate overestimation bias in Q-learning, the SAC agent utilizes two independent Q-networks (Twin Critics) and uses the minimum of the two Q-values for the Bellman target updates.

- **Implementation Check:** ✅
- **Notes:** Fully implemented inside the `DiscreteSAC` module. The Q-network class calculates two independent Q-values and takes `torch.min(q1, q2)` to evaluate policy loss.

---

## 5. Entropy-Based Novelty Detection
**Description in Paper:**
Zero-day and unknown attacks are initially flagged during inference if the output distribution of the SAC policy exhibits high entropy. A sample is flagged if its uncertainty crosses a mathematically defined threshold ($\tau$), indicating the agent does not recognize the flow as any of its known categories.

- **Implementation Check:** ⚠️
- **Notes:** The implementation diverges here for better performance. While an `entropy_detector.py` is available as legacy, the main pipeline in `evaluate_ids.py` strictly utilizes **Confidence Bounds** (max softmax thresholds) and **Mahalanobis Centroids** in the latent space as they proved far superior at isolating zero-day attacks (>80% UADR) compared to raw entropy distributions.

---

## 6. Unknown Buffer & Class Expansion
**Description in Paper:**
Traffic flagged as "Unknown" is captured in an isolated buffer. Periodically, a density-based cluster discovery algorithm (e.g., DBSCAN) analyzes the latent geometries of these unknown samples. Stable clusters are assigned new Class IDs, and the SAC network's action space is dynamically expanded to learn these new zero-day attacks continually.

- **Implementation Check:** ✅
- **Notes:** Fully implemented. `evaluate_ids.py` buffers unknowns into `UnknownBuffer` and invokes `cluster_discovery.py`, utilizing DBSCAN to identify valid clusters. The SAC's actor/critic output heads are resized natively on-the-fly (`sac.expand_action_space`).

---

## 7. Stability Regularization (EWC)
**Description in Paper:**
To prevent catastrophic forgetting when the model expands its architecture or updates on new intrusion distributions, Elastic Weight Consolidation (EWC) is applied to regularize the network parameter drift, maintaining performance on previously learned attacks.

- **Implementation Check:** ❌
- **Notes:** **Not Implemented.** While the action space gracefully expands upon discovering new classes, there is no Elastic Weight Consolidation (EWC) penalty constraint calculated or applied to the Loss gradients during `train_agent.py` backward passes to prevent catastrophic forgetting.

---

## 8. Datasets (CICIDS2017, TON-IoT)
**Description in Paper:**
The framework is evaluated rigorously against standard, highly-imbalanced cyber-security datasets such as CIC-IDS-2017 and TON-IoT.

- **Implementation Check:** ⚠️
- **Notes:** Evaluated on a combined `train_test_network.csv` feature set rather than downloading and mapping the distinct CICIDS/TON-IoT raw PCAP payloads. The core mathematical structures of the data are identical.

---

## 9. Data Preprocessing
**Description in Paper:**
Network frames undergo specific feature normalization (e.g., Min-Max or Standard Scaling) and categorical encoding before being shaped into sliding-window sequences tailored for the LSTM encoder.

- **Implementation Check:** ⚠️
- **Notes:** Implemented, but uses `StandardScaler` (Z-score normalization) instead of the paper's specified `MinMax` normalization to better serve the unbounded properties of the Mahalanobis Centroid distance calculations used later in the pipeline.

---

## 10. Training Hyperparameters
**Description in Paper:**
The paper defines specific learning rates, temperature coefficients ($\alpha$), reward bases, and sequence lengths vital to recreating the published results.

- **Implementation Check:** ✅
- **Notes:** Fully implemented exactly as described within `evaluate_ids.py`. Phase 1 rigorously reports Precision, Recall, F1, and FAR for known classes. Phase 2 discovers hidden distributions and outputs the Unknown Attack Detection Rate.

---

## 11. Evaluation Protocol
**Description in Paper:**
Performance is measured sequentially. Initial metrics evaluate known-class precision, recall, accuracy, and False Alarm Rates (Phase 1). Subsequently, metrics evaluate the system's ability to discover and isolate deliberately hidden zero-day classes, quantified as the Unknown Attack Detection Rate (Phase 2).

- **Implementation Check:** ✅
- **Notes:** Fully implemented exactly as described within `evaluate_ids.py`. Phase 1 rigorously reports Precision, Recall, F1, and FAR for known classes. Phase 2 discovers hidden distributions and outputs the Unknown Attack Detection Rate.

---

## 12. Continual Learning Scenario
**Description in Paper:**
The system does not retrain from scratch. It continuously adapts by updating the actor-critic weights iteratively using memory from the replay buffers as novel threats emerge.

- **Implementation Check:** ⚠️
- **Notes:** The architecture supports and allows arbitrary retraining over time using `UnknownBuffer` flushes and dynamic environment expansion, but does not yet contain an automated endless "Phase 3" continual training wrapper script. Expanding the model currently forces the user to manually trigger the next training cycle using the modified weights.
