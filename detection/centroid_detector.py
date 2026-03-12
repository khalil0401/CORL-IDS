import numpy as np
from scipy.spatial.distance import mahalanobis
import torch

class CentroidDetector:
    """
    Detects unknown attacks by computing the distance of the latent representation
    z_t to the known class centroids. Utilizes Mahalanobis distance if covariance
    is stable, otherwise falls back to Euclidean distance.
    """

    def __init__(self, distance_multiplier: float = 3.0):
        self.distance_multiplier = distance_multiplier
        self.centroids = {}
        self.covariances = {}
        self.inv_covariances = {}
        self.thresholds = {}
        self.use_mahalanobis = {}
        
    def fit(self, z_train: np.ndarray, y_train: np.ndarray, num_classes: int):
        """
        Compute centroids, covariances, and adaptive thresholds for each known class.
        """
        for cls in range(num_classes):
            idx = (y_train == cls)
            if not np.any(idx):
                continue
                
            z_cls = z_train[idx]
            mu = np.mean(z_cls, axis=0)
            self.centroids[cls] = mu
            
            # Try computing covariance for Mahalanobis
            if len(z_cls) > z_cls.shape[1]: 
                cov = np.cov(z_cls, rowvar=False)
                # Add small diagonal stabilization
                cov += np.eye(cov.shape[0]) * 1e-6
                
                try:
                    inv_cov = np.linalg.inv(cov)
                    self.use_mahalanobis[cls] = True
                    self.covariances[cls] = cov
                    self.inv_covariances[cls] = inv_cov
                    
                    # Compute training distances to set threshold
                    dists = np.array([mahalanobis(z, mu, inv_cov) for z in z_cls])
                except np.linalg.LinAlgError:
                    # Fallback to Euclidean
                    self.use_mahalanobis[cls] = False
                    dists = np.linalg.norm(z_cls - mu, axis=1)
            else:
                self.use_mahalanobis[cls] = False
                dists = np.linalg.norm(z_cls - mu, axis=1)
                
            mean_dist = np.mean(dists)
            std_dist = np.std(dists)
            self.thresholds[cls] = mean_dist + self.distance_multiplier * std_dist
            
    def predict_batch(self, z_batch: np.ndarray) -> np.ndarray:
        """
        Predict if a sample is UNKNOWN based on distance to the closest known centroid.
        """
        if not self.centroids:
            return np.zeros(len(z_batch), dtype=bool)
            
        N = len(z_batch)
        min_dists = np.full(N, np.inf)
        closest_cls_thresholds = np.full(N, np.inf)
        
        for i in range(N):
            z = z_batch[i]
            
            for cls, mu in self.centroids.items():
                if self.use_mahalanobis[cls]:
                    inv_cov = self.inv_covariances[cls]
                    try:
                        d = mahalanobis(z, mu, inv_cov)
                    except ValueError:
                        d = np.linalg.norm(z - mu)
                else:
                    d = np.linalg.norm(z - mu)
                    
                if d < min_dists[i]:
                    min_dists[i] = d
                    closest_cls_thresholds[i] = self.thresholds[cls]
                    
        is_unknown = min_dists > closest_cls_thresholds
        return is_unknown
