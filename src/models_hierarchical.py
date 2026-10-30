"""
Hierarchical 'Adaptive Family Gates' classifier for Kaggle 'Name That LLM'.
Two-stage pipeline:
1. Family Gate: Classifies passage into AI Lab family (OpenAI, Anthropic, Google, ZAI, Qwen, Moonshot).
2. Intra-Family Experts: Specialized classifiers trained on subtle intra-family variants.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from .config import (
    CANDIDATE_MODELS,
    FAMILY_NAMES,
    FAMILY_TO_IDX,
    FAMILY_TO_MODEL_INDICES,
    MODEL_FAMILY_MAP,
    MODEL_TO_IDX,
    NUM_CLASSES,
    NUM_FAMILIES,
    RANDOM_SEED,
)
from .models_tabular import CalibratedRidgeModel, softmax


class AdaptiveFamilyGateClassifier:
    """
    Hierarchical ensemble implementing Adaptive Family Gating:
    - Master Gate: Distinguishes between the 6 AI lab families.
    - Sub-Gates: Intra-family classifiers for labs with multiple candidate checkpoints.
    - Blends hierarchical probability P(Family) * P(Model | Family) with flat model probability.
    """

    def __init__(
        self,
        alpha_family: float = 1.0,
        alpha_model: float = 0.8,
        hierarchical_weight: float = 0.5,
    ):
        self.hierarchical_weight = hierarchical_weight

        # Flat 18-class classifier
        self.flat_classifier = CalibratedRidgeModel(alpha=alpha_model, temperature=1.0)

        # Level 1: Family Gate (6 classes)
        self.family_gate = CalibratedRidgeModel(alpha=alpha_family, temperature=1.0)

        # Level 2: Sub-gates for families with >1 model
        self.intra_family_experts: Dict[str, CalibratedRidgeModel] = {}
        for fam in FAMILY_NAMES:
            model_indices = FAMILY_TO_MODEL_INDICES[fam]
            if len(model_indices) > 1:
                self.intra_family_experts[fam] = CalibratedRidgeModel(
                    alpha=alpha_model * 0.8, temperature=1.0
                )

    def fit(self, X: Union[np.ndarray, sp.csr_matrix], y: np.ndarray):
        """
        Fits flat model, family gate, and intra-family experts.
        y: array of model class indices in [0, 17].
        """
        # 1. Fit flat model
        self.flat_classifier.fit(X, y)

        # 2. Derive family labels
        y_family = np.array([
            FAMILY_TO_IDX[MODEL_FAMILY_MAP[CANDIDATE_MODELS[idx]]] for idx in y
        ])
        self.family_gate.fit(X, y_family)

        # 3. Fit intra-family experts on family subsets
        for fam, expert in self.intra_family_experts.items():
            fam_idx = FAMILY_TO_IDX[fam]
            mask = y_family == fam_idx
            if np.sum(mask) >= len(FAMILY_TO_MODEL_INDICES[fam]):
                X_fam = X[mask]
                y_fam_raw = y[mask]

                # Map global model indices to local intra-family indices [0, num_models_in_fam - 1]
                local_map = {
                    global_idx: local_idx
                    for local_idx, global_idx in enumerate(FAMILY_TO_MODEL_INDICES[fam])
                }
                y_fam_local = np.array([local_map[g] for g in y_fam_raw])

                expert.fit(X_fam, y_fam_local)

        return self

    def predict_proba(self, X: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        """
        Computes blended probabilities:
        P_final = (1 - w) * P_flat + w * P_hierarchical
        where P_hierarchical(Model_k) = P(Family_f) * P(Model_k | Family_f)
        """
        N = X.shape[0]

        # 1. Flat probabilities (N, 18)
        P_flat = self.flat_classifier.predict_proba(X)

        # 2. Family gate probabilities (N, 6)
        P_family = self.family_gate.predict_proba(X)

        # 3. Hierarchical probabilities (N, 18)
        P_hier = np.zeros((N, NUM_CLASSES), dtype=np.float32)

        for fam_idx, fam in enumerate(FAMILY_NAMES):
            fam_prob = P_family[:, fam_idx : fam_idx + 1]  # (N, 1)
            model_indices = FAMILY_TO_MODEL_INDICES[fam]

            if len(model_indices) == 1:
                # Deterministic conditional prob = 1.0
                g_idx = model_indices[0]
                P_hier[:, g_idx] = fam_prob[:, 0]
            else:
                expert = self.intra_family_experts.get(fam)
                if expert is not None:
                    # Intra-family local probabilities (N, len(model_indices))
                    local_probs = expert.predict_proba(X)
                    for local_i, global_i in enumerate(model_indices):
                        P_hier[:, global_i] = (fam_prob * local_probs[:, local_i : local_i + 1])[:, 0]
                else:
                    # Uniform distribution among family members if expert not fitted
                    for global_i in model_indices:
                        P_hier[:, global_i] = (fam_prob / len(model_indices))[:, 0]

        # 4. Normalize P_hier
        row_sums = np.sum(P_hier, axis=1, keepdims=True)
        P_hier = P_hier / np.clip(row_sums, 1e-12, None)

        # 5. Blend flat and hierarchical
        w = self.hierarchical_weight
        P_blended = (1.0 - w) * P_flat + w * P_hier
        return P_blended

    def predict(self, X: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
