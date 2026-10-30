"""
Tabular and linear models module for Kaggle 'Name That LLM'.
Implements calibrated Ridge, Logistic Regression, SGD, and HistGradientBoosting classifiers.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.preprocessing import StandardScaler

from .config import NUM_CLASSES, RANDOM_SEED


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class CalibratedRidgeModel:
    """
    Ridge Classifier with temperature-calibrated softmax probability output.
    Extremely fast and highly competitive on high-dimensional text + stylometric features.
    """

    def __init__(self, alpha: float = 1.0, temperature: float = 1.0):
        self.alpha = alpha
        self.temperature = temperature
        self.clf = RidgeClassifier(
            alpha=alpha,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="auto",
        )

    def fit(self, X: Union[np.ndarray, sp.csr_matrix], y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        decision = self.clf.decision_function(X)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        return softmax(decision / self.temperature, axis=1)

    def predict(self, X: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class CalibratedLogisticModel:
    """
    Regularized Multinomial Logistic Regression with balanced class weights.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 200):
        self.C = C
        self.max_iter = max_iter
        self.clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )

    def fit(self, X: Union[np.ndarray, sp.csr_matrix], y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        return self.clf.predict(X)


class FastSGDModel:
    """
    Stochastic Gradient Descent classifier with modified Huber loss
    (outputs well-calibrated probabilities, robust to outliers).
    """

    def __init__(self, alpha: float = 1e-4, max_iter: int = 1000):
        self.clf = SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=alpha,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=RANDOM_SEED,
        )

    def fit(self, X: Union[np.ndarray, sp.csr_matrix], y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: Union[np.ndarray, sp.csr_matrix]) -> np.ndarray:
        return self.clf.predict(X)


class GBDTStylometryModel:
    """
    Histogram-based Gradient Boosting Decision Tree trained on dense stylometric features.
    Captures non-linear typographic, punctuation, and structural feature interactions.
    """

    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 10,
    ):
        self.scaler = StandardScaler()
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )

    def fit(self, X_dense: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_dense)
        self.clf.fit(X_scaled, y)
        return self

    def predict_proba(self, X_dense: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_dense)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X_dense: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X_dense), axis=1)
