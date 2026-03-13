import numpy as np
import torch

class ConfidenceUnknownDetector:
    """
    Detects unknown attacks by analyzing the maximum softmax confidence probability
    output by the SAC actor. If the confidence is below the threshold, the sample
    is flagged as anomalous (UNKNOWN).
    """

    def __init__(self, beta: float = 1.0, default_threshold: float = 0.90):
        self.beta = beta
        self.threshold = default_threshold
        self.fitted = False

    def fit(self, probs: np.ndarray):
        """
        Dynamically calculate the threshold based on the confidence distribution
        of the known training dataset: threshold = mean - (beta * std).
        """
        confidences = np.max(probs, axis=-1)
        mean_conf   = np.mean(confidences)
        std_conf    = np.std(confidences)

        # Ensure the adaptive threshold doesn't exceed 1.0 or drop unreasonably low
        self.threshold = float(np.clip(mean_conf - (self.beta * std_conf), 0.50, 0.99))
        self.fitted    = True
        print(f"    -> [CONFIDENCE] Adaptive Threshold Fitted: {self.threshold:.3f} (mean={mean_conf:.3f}, std={std_conf:.3f})")

    def predict_batch(self, probs: np.ndarray) -> np.ndarray:
        """
        Flag samples where max probability < threshold.
        
        Parameters
        ----------
        probs : np.ndarray
            Shape (batch_size, num_actions)
            
        Returns
        -------
        is_unknown : np.ndarray (batch_size,) of booleans
        """
        confidences = np.max(probs, axis=-1)
        is_unknown = confidences < self.threshold
        return is_unknown
