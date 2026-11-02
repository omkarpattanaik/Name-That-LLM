"""
Shared-Prompt Two-Way Gaussian Decomposition (WPCA + Ledoit-Wolf Shrinkage).
Implements the exact two-way random-effects LDA model on multi-layer embeddings:
Z = mu[y] + P[prompt] + E
with analytical Ledoit-Wolf covariance shrinkage and Cholesky joint whitening.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin


def _lw_shrinkage(X_residuals: np.ndarray, dof: Optional[int] = None) -> np.ndarray:
    """
    Computes analytical Ledoit-Wolf shrinkage toward scaled identity matrix:
    S_shrunk = (1 - lambda) * S + lambda * (tr(S)/d) * I
    """
    K, d = X_residuals.shape
    denom = dof if dof is not None and dof > 0 else max(1, K)
    S = (X_residuals.T @ X_residuals) / denom
    nu = float(np.trace(S)) / float(d)
    T = nu * np.eye(d, dtype=S.dtype)

    delta2 = float(np.sum((S - T) ** 2))
    if delta2 < 1e-12:
        return S + 1e-4 * np.eye(d, dtype=S.dtype)

    # Sample variance of empirical covariance elements
    phi = float(np.sum((X_residuals.T @ X_residuals) ** 2) / denom - np.sum(S ** 2))
    gamma = max(0.0, min(phi / (delta2 * denom), 1.0))
    S_shrunk = (1.0 - gamma) * S + gamma * T + 1e-4 * np.eye(d, dtype=S.dtype)
    return S_shrunk


class SharedPromptGaussianDecomposer(BaseEstimator, ClassifierMixin):
    """
    Two-Way Random-Effects Latent Gaussian Model:
    Z = mu[class] + P[prompt] + E
    Features are projected via Within-Prompt PCA (WPCA) to rank d=128,
    fitted via 4 iterations of alternating means,
    and whitened via Cholesky decomposition of Ledoit-Wolf regularized within-prompt covariance.
    """

    def __init__(self, n_components: int = 128, n_iter: int = 4):
        self.n_components = n_components
        self.n_iter = n_iter

        self.mu_X_ = None
        self.sigma_X_ = None
        self.V_ = None
        self.mu_ = None
        self.W_ = None
        self.mu_w_ = None
        self.classes_ = None
        self.class_priors_ = None

    def fit(self, X: np.ndarray, y: Union[List, np.ndarray, pd.Series], groups: Union[List, np.ndarray, pd.Series]):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)
        groups_arr = np.asarray(groups)

        N, D = X_arr.shape
        self.classes_ = np.unique(y_arr)
        C = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y_arr])

        unique_groups = np.unique(groups_arr)
        G = len(unique_groups)
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        g_idx = np.array([group_to_idx[g] for g in groups_arr])

        # 1. Standardize X -> Xs
        self.mu_X_ = np.mean(X_arr, axis=0, keepdims=True)
        self.sigma_X_ = np.std(X_arr, axis=0, keepdims=True) + 1e-6
        Xs = (X_arr - self.mu_X_) / self.sigma_X_

        # 2. Within-Prompt PCA (wpca)
        Pm = np.zeros((G, D), dtype=np.float32)
        for g in range(G):
            mask_g = (g_idx == g)
            if mask_g.any():
                Pm[g] = np.mean(Xs[mask_g], axis=0)

        R = Xs - Pm[g_idx]
        d = min(self.n_components, D, N - 1)
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        self.V_ = Vt[:d].T.astype(np.float32)
        Z = Xs @ self.V_

        # 3. Additive Model Fit (4 alternating iterations): Z = mu[y] + P[g] + E
        mu = np.zeros((C, d), dtype=np.float32)
        P = np.zeros((G, d), dtype=np.float32)

        for _ in range(self.n_iter):
            # Update class centroids mu[c]
            for c in range(C):
                mask_c = (y_idx == c)
                if mask_c.any():
                    mu[c] = np.mean(Z[mask_c] - P[g_idx[mask_c]], axis=0)
            # Update prompt effects P[g]
            for g in range(G):
                mask_g = (g_idx == g)
                if mask_g.any():
                    P[g] = np.mean(Z[mask_g] - mu[y_idx[mask_g]], axis=0)

        self.mu_ = mu
        E = Z - mu[y_idx] - P[g_idx]

        # 4. Within-prompt error covariance Se and Between-prompt Sp
        dof = max(1, N - C - G)
        Se = _lw_shrinkage(E, dof=dof)
        Sp = _lw_shrinkage(P, dof=max(1, G))

        # 5. Cholesky Joint Diagonalization / Whitening
        L = cholesky(Se, lower=True)
        self.W_ = solve_triangular(L, np.eye(d, dtype=np.float32), lower=True).astype(np.float32)
        self.mu_w_ = (self.mu_ @ self.W_.T).astype(np.float32)

        # Class priors
        counts = np.bincount(y_idx, minlength=C)
        self.class_priors_ = np.log(np.maximum(counts / float(N), 1e-9)).astype(np.float32)
        print(f"SharedPromptGaussianDecomposer fitted: N={N}, D={D} -> d={d}, C={C}, G={G}, dof={dof}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        # Standardize and project
        Xs = (X_arr - self.mu_X_) / self.sigma_X_
        Z = Xs @ self.V_
        # Whiten
        Zw = Z @ self.W_.T

        # Whitened distance to class centroids: ||Zw - mu_w[c]||^2
        # Zw is (M, d), mu_w is (C, d) -> dists is (M, C)
        dists = np.sum((Zw[:, None, :] - self.mu_w_[None, :, :]) ** 2, axis=2)
        logits = -0.5 * dists + self.class_priors_[None, :]
        probs = softmax(logits, axis=1)
        return probs.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
