"""
Post-processing and probability calibration module for Kaggle 'Name That LLM'.
Implements Sinkhorn-Knopp optimal transport marginal balancing, temperature scaling,
and log-probability cluster adjustments to directly optimize Balanced Accuracy.
"""

from typing import Optional
import numpy as np


def sinkhorn_knopp_balancing(
    probs: np.ndarray,
    target_prior: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Applies the Sinkhorn-Knopp algorithm to calibrate the predicted probability matrix P.
    Normalizes rows (each sample's probabilities sum to 1) and columns (each class's
    expected frequency matches the target prior, default uniform 1/K for balanced accuracy).

    Args:
        probs: (N, K) predicted probability matrix.
        target_prior: (K,) desired target marginal distribution. Defaults to uniform (1/K).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        eps: Epsilon for numerical stability.

    Returns:
        Calibrated (N, K) probability matrix.
    """
    N, K = probs.shape
    if target_prior is None:
        target_prior = np.ones(K, dtype=np.float64) / K
    else:
        target_prior = np.asarray(target_prior, dtype=np.float64)
        target_prior = target_prior / np.sum(target_prior)

    # Work in double precision
    P = np.clip(probs.astype(np.float64), eps, 1.0)
    # Target row sums (1.0 for each sample)
    r = np.ones(N, dtype=np.float64)
    # Target column sums (N * prior_k)
    c = N * target_prior

    u = np.ones(N, dtype=np.float64)
    v = np.ones(K, dtype=np.float64)

    for _ in range(max_iter):
        # Update u (scale rows)
        denom_u = P @ v
        denom_u = np.clip(denom_u, eps, None)
        u_new = r / denom_u

        # Update v (scale columns)
        denom_v = P.T @ u_new
        denom_v = np.clip(denom_v, eps, None)
        v_new = c / denom_v

        # Check convergence
        diff = np.max(np.abs(u_new - u)) + np.max(np.abs(v_new - v))
        u = u_new
        v = v_new
        if diff < tol:
            break

    # Balanced probability matrix
    P_balanced = (u[:, None] * P) * v[None, :]
    # Normalize rows to sum to 1
    row_sums = np.sum(P_balanced, axis=1, keepdims=True)
    P_balanced = P_balanced / np.clip(row_sums, eps, None)

    return P_balanced.astype(np.float32)


def temperature_scaled_adjustment(
    probs: np.ndarray,
    temperature: float = 1.0,
    prior_weight: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Adjusts raw probabilities by temperature and class prior logit adjustment:
    z_ik = (log(p_ik) / T) - prior_weight * log(mean_class_freq)
    """
    P = np.clip(probs, eps, 1.0)
    log_p = np.log(P)

    # Empirical class frequencies
    mean_freq = np.mean(P, axis=0, keepdims=True)
    mean_freq = np.clip(mean_freq, eps, None)

    # Scaled logits
    logits = (log_p / max(1e-4, temperature)) - prior_weight * np.log(mean_freq)

    # Softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    return (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).astype(np.float32)


def soft_prompt_cluster_centering(
    probs: np.ndarray,
    prompt_ids: np.ndarray,
    alpha: float = 0.15,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Softly penalizes over-represented classes within estimated prompt clusters
    WITHOUT enforcing brittle hard 1-to-1 Hungarian assignments.

    Args:
        probs: (N, K) probabilities.
        prompt_ids: (N,) array of cluster or prompt identifiers.
        alpha: Penalty strength.
    """
    adjusted_probs = probs.copy()
    unique_prompts = np.unique(prompt_ids)

    for pid in unique_prompts:
        mask = prompt_ids == pid
        if np.sum(mask) <= 1:
            continue

        cluster_p = probs[mask]
        # Cluster mean probability per class
        cluster_mean = np.mean(cluster_p, axis=0, keepdims=True)
        # Log-probability adjustment
        log_p = np.log(np.clip(cluster_p, eps, 1.0))
        # Subtract soft fraction of cluster average to prevent multiple identical models
        adj_log_p = log_p - alpha * np.log(np.clip(cluster_mean, eps, 1.0))
        # Softmax back to probability
        exp_p = np.exp(adj_log_p - np.max(adj_log_p, axis=1, keepdims=True))
        adjusted_probs[mask] = exp_p / np.sum(exp_p, axis=1, keepdims=True)

    return adjusted_probs
