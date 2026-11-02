"""
Multi-Layer Intermediate Transformer Pooling and Twin Readout Heads.
Extracts intermediate hidden states to isolate stylometric signal from prompt semantics,
and provides prompt-centered twin classification.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def group_center(X: np.ndarray, groups: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
    """
    Subtracts group-wise mean from each sample's feature vector.
    Eliminates prompt-topic confounding in representation space.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    groups_arr = np.asarray(groups)
    centered = np.empty_like(X_arr)

    unique_groups = np.unique(groups_arr)
    for g in unique_groups:
        mask = (groups_arr == g)
        if mask.any():
            g_mean = np.mean(X_arr[mask], axis=0, keepdims=True)
            centered[mask] = X_arr[mask] - g_mean

    return centered


class IntermediateLayerPooler:
    """
    Extracts and pools intermediate hidden states across transformer layers.
    - Qwen3: layers [2, 4, 6, 8, 12, 16, 20, 24]
    - DeBERTa-v3-large: layers [6, 12, 16, 18, 20, 21, 22, 23, 24]
    """

    DEBERTA_LAYERS = [6, 12, 16, 18, 20, 21, 22, 23, 24]
    QWEN_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]

    def __init__(self, target_layers: Optional[List[int]] = None, eps: float = 1e-5):
        self.target_layers = target_layers or self.DEBERTA_LAYERS
        self.eps = eps

    def pool_hidden_states(
        self,
        hidden_states: Tuple[np.ndarray, ...],
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Mean pools tokens using attention mask for each target layer,
        applies LayerNorm, and concatenates along feature dimension.
        """
        pooled_layers = []
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        mask_sum = np.maximum(mask_expanded.sum(axis=1), 1e-9)

        for l_idx in self.target_layers:
            if l_idx < len(hidden_states):
                h = hidden_states[l_idx]
                # Mean pool
                mean_pooled = (h * mask_expanded).sum(axis=1) / mask_sum
                # LayerNorm (zero mean, unit variance per sample)
                u = mean_pooled.mean(axis=-1, keepdims=True)
                s = mean_pooled.std(axis=-1, keepdims=True)
                ln_pooled = (mean_pooled - u) / (s + self.eps)
                pooled_layers.append(ln_pooled)

        if not pooled_layers:
            # Fallback to last available layer
            h = hidden_states[-1]
            return ((h * mask_expanded).sum(axis=1) / mask_sum).astype(np.float32)

        return np.concatenate(pooled_layers, axis=-1).astype(np.float32)


class TwinPromptCenteredClassifier:
    """
    Fits both a plain 18-way readout head on raw pooled features E_tr
    and a prompt-centered twin head trained on group_center(E_tr, prompt_id).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler_plain = StandardScaler()
        self.clf_plain = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )

        self.scaler_center = StandardScaler()
        self.clf_center = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Union[List, np.ndarray, pd.Series]):
        # Plain head
        X_sc_plain = self.scaler_plain.fit_transform(X)
        self.clf_plain.fit(X_sc_plain, y)

        # Centered twin head
        X_cent = group_center(X, groups)
        X_sc_cent = self.scaler_center.fit_transform(X_cent)
        self.clf_center.fit(X_sc_cent, y)
        return self

    def predict_proba_plain(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler_plain.transform(X)
        return self.clf_plain.predict_proba(X_sc)

    def predict_proba_center(self, X_centered: np.ndarray) -> np.ndarray:
        X_sc = self.scaler_center.transform(X_centered)
        return self.clf_center.predict_proba(X_sc)
