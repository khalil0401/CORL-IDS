import numpy as np
import torch

class ConfidenceUnknownDetector:
    """
    Detects unknown attacks by analyzing the maximum softmax confidence probability
    output by the SAC actor. If the confidence is below the threshold, the sample
    is flagged as anomalous (UNKNOWN).
    """

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

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
