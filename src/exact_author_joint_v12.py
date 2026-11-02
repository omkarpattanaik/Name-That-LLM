import itertools
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ==============================================================================
# 1. Size-Capped Agglomerative Prompt Recovery on Content-Word TF-IDF
# ==============================================================================
def recover_test_prompts_size_capped(
    texts: pd.Series,
    sim_threshold: float = 0.26,
    max_cluster_size: int = 18
) -> np.ndarray:
    """
    Recovers shared-prompt groups via greedy UPGMA average-linkage merging
    with a strict cluster size cap (size[i] + size[j] <= 18) so topical clusters
    never snowball into mega-clusters.
    """
    n = len(texts)
    if n < 2:
        return np.arange(n, dtype=int)

    try:
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
            max_features=60000
        )
        X = vec.fit_transform(texts.astype(str))
        A = (X @ X.T).toarray().astype(np.float32)
        np.fill_diagonal(A, -np.inf)
    except Exception:
        return np.arange(n, dtype=int)

    size = np.ones(n, dtype=int)
    members = [[i] for i in range(n)]
    active = np.ones(n, dtype=bool)

    while True:
        ok = (size[:, None] + size[None, :] <= max_cluster_size) & active[:, None] & active[None, :]
        np.fill_diagonal(ok, False)
        M = np.where(ok, A, -np.inf)
        k = int(np.argmax(M))
        i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < sim_threshold:
            break

        # Weighted UPGMA average similarity update
        new_sim = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new_sim
        A[:, i] = new_sim
        A[i, i] = -np.inf
        A[i, j] = -np.inf
        A[j, i] = -np.inf
        A[j, :] = -np.inf
        A[:, j] = -np.inf

        size[i] += size[j]
        members[i].extend(members[j])
        members[j] = []
        active[j] = False

    labels = np.empty(n, dtype=int)
    cid = 0
    for i in range(n):
        if active[i] and members[i]:
            for idx in members[i]:
                labels[idx] = cid
            cid += 1
    return labels


# ==============================================================================
# 2. Exact Shared-Prompt Gaussian Cavity Inversion & Uniqueness Posteriors
# ==============================================================================
def _log_softmax(L: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(L, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (L - m) - np.log(np.sum(np.exp(L - m), axis=axis, keepdims=True))

def _inj_mask(n: int, C: int) -> np.ndarray:
    """Boolean mask of shape (C,)*n that is True wherever any two indices are equal."""
    grid = np.indices((C,) * n)
    mask = np.zeros((C,) * n, dtype=bool)
    for i, j in itertools.combinations(range(n), 2):
        mask |= (grid[i] == grid[j])
    return mask

def exact_tuple_marginals(Q: np.ndarray, T: float = 1.0, inj: bool = True) -> np.ndarray:
    """
    Computes exact per-passage log marginal posteriors (n, C) from an n-way
    tuple log-joint table Q, masking out duplicate authors when inj=True.
    Used for small clusters (2 <= n <= 3).
    """
    n = Q.ndim
    C = Q.shape[0]
    Qt = Q / T
    if inj and n >= 2:
        Qt = np.where(_inj_mask(n, C), -np.inf, Qt)

    out = np.zeros((n, C), dtype=np.float64)
    for i in range(n):
        axes = tuple(ax for ax in range(n) if ax != i)
        m = np.max(Qt, axis=axes, keepdims=True)
        m_sq = np.where(np.isfinite(m), m, 0.0)
        log_marg = np.squeeze(np.log(np.sum(np.exp(Qt - m_sq), axis=axes, keepdims=True)) + m_sq, axis=axes)
        out[i] = _log_softmax(log_marg, axis=-1)
    return out

def singleton_ll(U: np.ndarray, m: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Log-likelihood of whitened features U (n, d) under marginal variance (1 + lam)."""
    diff = U[:, None, :] - m[None, :, :]  # (n, C, d)
    var = 1.0 + lam[None, None, :]
    return -0.5 * np.sum((diff ** 2) / var + np.log(var), axis=-1)

def cavity_group_logpost(
    U: np.ndarray,
    m: np.ndarray,
    lam: np.ndarray,
    T1: float = 1.0,
    Tg: float = 1.0,
    iters: int = 5,
    repel: bool = True,
    exact_max_n: int = 3
) -> np.ndarray:
    """
    Computes shared-prompt Gaussian posteriors for a prompt group U (n, d):
      - Subtract expected model style (U - q @ m) to isolate pure prompt residual.
      - Estimate leave-one-out cavity prompt-effect mean Pm with Wiener shrinkage v.
      - Apply exact tuple marginalization for n <= exact_max_n, or iterative
        cavity mean-field at-most-once repulsion for larger clusters.
    """
    n, d = U.shape
    C = m.shape[0]

    # Initialize q from singleton posteriors
    L0 = singleton_ll(U, m, lam) / T1
    q = np.exp(_log_softmax(L0, axis=-1))

    # Bayesian Wiener shrinkage factor v_d = lam_d / (1 + (n - 1) * lam_d)
    v = lam / (1.0 + (n - 1.0) * lam)  # (d,)

    if n <= exact_max_n:
        # Construct exact n-way joint log-likelihood tensor Q over (C,)*n
        # Marginalizing over the shared Gaussian prompt prior P ~ N(0, diag(lam))
        inv_var = 1.0 - (lam / (1.0 + n * lam))  # diagonal weight
        cross_var = lam / (1.0 + n * lam)        # off-diagonal pair weight

        U_res = U[:, None, :] - m[None, :, :]    # (n, C, d)
        unary = -0.5 * np.sum((U_res ** 2) * inv_var[None, None, :], axis=-1)  # (n, C)

        Q = np.zeros((C,) * n, dtype=np.float64)
        for i in range(n):
            sh = [1] * n
            sh[i] = C
            Q = Q + unary[i].reshape(sh)

        for i, j in itertools.combinations(range(n), 2):
            # Cross-term between passage i (class c1) and passage j (class c2)
            B_ij = np.einsum("cd,kd,d->ck", U_res[i], U_res[j], cross_var)
            sh = [1] * n
            sh[i] = C
            sh[j] = C
            Q = Q + B_ij.reshape(sh)

        return exact_tuple_marginals(Q, T=Tg, inj=repel)

    # Iterative Cavity Prompt-Effect Inversion for n > exact_max_n
    for _ in range(iters):
        s = U - (q @ m)                       # Style-decoupled prompt residuals (n, d)
        S = s.sum(axis=0, keepdims=True)      # Total prompt sum (1, d)
        Pm = (S - s) * v[None, :]             # Cavity prompt-effect means (n, d)

        R = U[:, None, :] - Pm[:, None, :] - m[None, :, :]  # (n, C, d)
        L = -0.5 * np.sum((R ** 2) / (1.0 + v[None, None, :]), axis=-1) / Tg

        if repel:
            # Mean-field at-most-once: discourage classes held by other cluster members
            lo = np.log(np.clip(1.0 - q, 1e-6, 1.0))
            L = L + (lo.sum(axis=0, keepdims=True) - lo)

        q = np.exp(_log_softmax(L, axis=-1))

    return _log_softmax(L, axis=-1)


# ==============================================================================
# 3. Margin-Adaptive Confidence-Gated Soft Sinkhorn Calibration
# ==============================================================================
def soft_sinkhorn(p: np.ndarray, strength: float = 0.78, iters: int = 70) -> np.ndarray:
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 1e-12)
    p /= p.sum(axis=1, keepdims=True)

    prior = (1.0 - strength) * p.mean(axis=0) + strength / p.shape[1]
    q = p.copy()
    for _ in range(iters):
        q *= prior / np.maximum(q.mean(axis=0), 1e-12)
        q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
    return q

def margin_adaptive_sinkhorn_postprocess(
    probs: np.ndarray,
    sinkhorn_strength: float = 0.78,
    base_mix: float = 0.64,
    low_margin_boost: float = 0.24,
    margin_scale: float = 0.35
) -> np.ndarray:
    """
    Applies soft Sinkhorn calibration with margin-adaptive log-interpolation:
    high-margin predictions retain their sharp local posteriors while uncertain
    boundary passages receive stronger global prior regularization.
    """
    n, k = probs.shape
    if n < 18:
        return probs

    p = np.maximum(probs, 1e-12)
    p /= p.sum(axis=1, keepdims=True)
    q = soft_sinkhorn(p, strength=sinkhorn_strength, iters=70)

    sorted_p = np.sort(p, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]

    logp = np.log(p)
    logq = np.log(q)

    # Adaptive interpolation weight per row based on top-2 margin
    row_strength = base_mix + low_margin_boost * np.maximum(0.0, 1.0 - margin / margin_scale)
    mix_logits = (1.0 - row_strength[:, None]) * logp + row_strength[:, None] * logq

    exp_mix = np.exp(mix_logits - mix_logits.max(axis=1, keepdims=True))
    return exp_mix / exp_mix.sum(axis=1, keepdims=True)
