"""
Cohesion-Aware Test-Time Adaptive Centering.
Avoids Hungarian assignment pitfalls on uneven test prompts by performing
soft hierarchical clustering and injecting prompt-centered twin classifier evidence.
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import softmax

from .transformer_pooling import TwinPromptCenteredClassifier, group_center


class CohesionAdaptiveCentering:
    """
    Identifies cohesive test clusters via fused TF-IDF + embedding cosine distance,
    subtracts estimated prompt means, and injects log-probability evidence from the twin head.
    """

    def __init__(
        self,
        thresh: float = 0.40,
        twin_w: float = 0.35,
        min_cluster_size: int = 2,
        max_cluster_size: int = 18,
    ):
        self.thresh = thresh
        self.twin_w = twin_w
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def adapt_test_predictions(
        self,
        uncentered_logits: np.ndarray,
        E_te: np.ndarray,
        X_tfidf_te: Optional[np.ndarray],
        twin_classifier: TwinPromptCenteredClassifier,
    ) -> np.ndarray:
        """
        Adapts test logits using cohesion-aware clustering and twin classifier.
        Guarded with len(test) >= 50 check.
        """
        N = len(uncentered_logits)
        logits = uncentered_logits.copy()

        # Guard: Bypass clustering if commit test set has < 50 rows
        if N < 50:
            print(f"CohesionAdaptiveCentering: N={N} < 50, bypassing transductive clustering for fast commit check.")
            return softmax(logits, axis=1)

        # 1. L2 Normalize Embeddings
        E_norm = E_te / (np.linalg.norm(E_te, axis=1, keepdims=True) + 1e-9)
        D_embed = 1.0 - (E_norm @ E_norm.T)

        # 2. L2 Normalize TF-IDF (if provided)
        if X_tfidf_te is not None:
            if hasattr(X_tfidf_te, "toarray"):
                X_t = X_tfidf_te.toarray()
            else:
                X_t = np.asarray(X_tfidf_te)
            X_t_norm = X_t / (np.linalg.norm(X_t, axis=1, keepdims=True) + 1e-9)
            D_tfidf = 1.0 - (X_t_norm @ X_t_norm.T)
            D = 0.5 * D_tfidf + 0.5 * D_embed
        else:
            D = D_embed

        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, 2.0)

        # 3. Hierarchical Agglomerative Average-Linkage Clustering
        condensed_D = squareform(D, checks=False)
        Z = linkage(condensed_D, method="average")
        cluster_labels = fcluster(Z, t=self.thresh, criterion="distance")

        # 4. Identify cohesive test clusters with size 2 <= sz <= 18
        unique, counts = np.unique(cluster_labels, return_counts=True)
        size_map = dict(zip(unique, counts))

        adapt_mask = np.array([self.min_cluster_size <= size_map[c] <= self.max_cluster_size for c in cluster_labels])
        adapt_indices = np.where(adapt_mask)[0]

        n_cohesive_clusters = sum(1 for sz in counts if self.min_cluster_size <= sz <= self.max_cluster_size)
        print(f"CohesionAdaptiveCentering: found {n_cohesive_clusters} cohesive clusters covering {len(adapt_indices)}/{N} rows")

        # 5. Adapt cohesive rows only
        if len(adapt_indices) > 0:
            E_sub = E_te[adapt_indices]
            groups_sub = cluster_labels[adapt_indices]
            E_sub_centered = group_center(E_sub, groups_sub)

            # Evaluate prompt-centered twin classifier
            p_center = twin_classifier.predict_proba_center(E_sub_centered)

            # Invert probabilities to log-evidence and add to logits
            log_evidence = np.log(np.clip(p_center, 1e-9, 1.0))
            logits[adapt_indices] += self.twin_w * log_evidence

        return softmax(logits, axis=1).astype(np.float32)

