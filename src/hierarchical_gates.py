"""
Hierarchical Specialist Sub-Family Gates.
Resolves intra-lab sibling confusion and straight-apostrophe ZAI vs Opus-5 ambiguity
via geometric probability redistribution preserving primary family probability mass.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FAMILY_CLAUDE = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
]

FAMILY_GEMINI = [
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

FAMILY_ZAI_OPUS = [
    "glm-5.3",
    "glm-5.3-flash",
    "claude-opus-5",
]


class SpecialistSubFamilyGate:
    """
    Second-stage gating classifier focused strictly on a subset of confused siblings.
    """

    def __init__(self, family_name: str, candidate_classes: List[str], C: float = 1.0, random_state: int = 42):
        self.family_name = family_name
        self.candidate_classes = list(candidate_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.candidate_classes)}
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=1000, random_state=random_state, solver="lbfgs")
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: Union[List[str], np.ndarray, pd.Series]):
        y_arr = np.asarray(y)
        mask = np.isin(y_arr, self.candidate_classes)
        if mask.sum() == 0:
            print(f"Warning: No training samples found for gate '{self.family_name}'")
            return self

        X_sub = X[mask]
        y_sub = np.array([self.class_to_idx[c] for c in y_arr[mask]])

        X_sc = self.scaler.fit_transform(X_sub)
        self.clf.fit(X_sc, y_sub)
        self.is_fitted = True
        print(f"SpecialistGate '{self.family_name}': fitted on {len(X_sub)} samples across {len(self.candidate_classes)} classes.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            # Uniform fallback
            n_classes = len(self.candidate_classes)
            return np.full((len(X), n_classes), 1.0 / n_classes, dtype=np.float32)

        X_sc = self.scaler.transform(X)
        probs = self.clf.predict_proba(X_sc)
        # Ensure correct column ordering matching self.candidate_classes
        if len(self.clf.classes_) < len(self.candidate_classes):
            full_probs = np.zeros((len(X), len(self.candidate_classes)), dtype=np.float32)
            for i, c_idx in enumerate(self.clf.classes_):
                full_probs[:, c_idx] = probs[:, i]
            return full_probs
        return probs.astype(np.float32)


class HierarchicalGatesEnsemble:
    """
    Manages the 3 specialist sub-family gates:
    1. 5-Way Anthropic Claude Gate
    2. 4-Way Google Gemini Gate
    3. 3-Way ZAI vs. Opus-5 Disambiguator
    """

    def __init__(self, alpha: float = 0.35, all_classes: Optional[List[str]] = None):
        self.alpha = alpha
        self.all_classes = all_classes or [
            "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
            "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
            "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
            "gpt-6-astra", "kimi-k3", "qwen3.8-max"
        ]
        self.class_to_idx18 = {c: i for i, c in enumerate(self.all_classes)}

        self.gate_claude = SpecialistSubFamilyGate("Claude-5Way", FAMILY_CLAUDE)
        self.gate_gemini = SpecialistSubFamilyGate("Gemini-4Way", FAMILY_GEMINI)
        self.gate_zai_opus = SpecialistSubFamilyGate("ZAI-Opus-3Way", FAMILY_ZAI_OPUS)

    def fit_all(self, X: np.ndarray, y: Union[List[str], np.ndarray, pd.Series]):
        self.gate_claude.fit(X, y)
        self.gate_gemini.fit(X, y)
        self.gate_zai_opus.fit(X, y)
        return self

    def apply_gate_blend(
        self,
        P_18: np.ndarray,
        gate: SpecialistSubFamilyGate,
        X_feats: np.ndarray,
    ) -> np.ndarray:
        """
        Redistributes probability mass inside gate's sibling family using weighted geometric blend (alpha=0.35).
        Preserves total family probability mass: sum_{c in family} P_blended[c] == sum_{c in family} P_18[c].
        """
        if not gate.is_fitted:
            return P_18

        P_res = P_18.copy()
        family_indices = [self.class_to_idx18[c] for c in gate.candidate_classes if c in self.class_to_idx18]
        if len(family_indices) <= 1:
            return P_res

        P_gate = gate.predict_proba(X_feats)

        for i in range(len(P_res)):
            # 1. Total family mass from 18-way ensemble
            m_family = float(P_res[i, family_indices].sum())
            if m_family < 1e-6:
                continue

            # 2. 18-way conditional probabilities within family
            p_cond_18 = P_res[i, family_indices] / m_family

            # 3. Gate probabilities
            p_g = P_gate[i]

            # 4. Weighted geometric blend: p_cond^(1-alpha) * p_gate^alpha
            p_geom = (np.maximum(p_cond_18, 1e-9) ** (1.0 - self.alpha)) * (np.maximum(p_g, 1e-9) ** self.alpha)

            # 5. Renormalize within family simplex
            p_geom_norm = p_geom / max(float(p_geom.sum()), 1e-9)

            # 6. Re-scale by total family mass
            P_res[i, family_indices] = m_family * p_geom_norm

        return P_res

    def refine_probabilities(self, P_18: np.ndarray, X_feats: np.ndarray) -> np.ndarray:
        """Applies Claude, Gemini, and ZAI-Opus gates sequentially."""
        P = self.apply_gate_blend(P_18, self.gate_claude, X_feats)
        P = self.apply_gate_blend(P, self.gate_gemini, X_feats)
        P = self.apply_gate_blend(P, self.gate_zai_opus, X_feats)
        return P

