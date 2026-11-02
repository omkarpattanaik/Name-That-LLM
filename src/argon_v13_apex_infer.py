import gc
import json
import os
import re
import hashlib
import numpy as np
import pandas as pd
try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

# Mandatory offline & T4 fp32-accumulator safeguards
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_HUB_KERNELS"] = "NO"
if HAS_TORCH and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# ==============================================================================
# 1. Stylometric & 4-View Preprocessors
# ==============================================================================
_FW = set(
    "a an the and or but if while although because as so than that which who whom whose "
    "what where when how why this these those it they we you i he she them us me him her "
    "my your his their our its is are was were be been being have has had do does did "
    "will would shall should can could may might must ought to of in on at by for with "
    "from into during including until against among throughout despite towards upon "
    "concerning about above across after along around before behind below beneath beside "
    "between beyond down except inside near off out outside over past since through "
    "under underneath up within without not no nor very too also just only even still "
    "already always never often sometimes usually however therefore moreover furthermore "
    "consequently nevertheless nonetheless otherwise instead meanwhile".split()
)

def shape_text(s: str) -> str:
    s = re.sub(r"[a-z]+", lambda m: "a" * min(len(m.group(0)), 4), str(s))
    s = re.sub(r"[A-Z]+", lambda m: "A" * min(len(m.group(0)), 3), s)
    return re.sub(r"[0-9]+", "0", s)

def small_words_text(s: str) -> str:
    tokens = re.findall(r"(?u)\b\w+\b|[^\w\s]", str(s))
    return " ".join(t if (not t.isalnum() or t.lower() in _FW or len(t) <= 3) else "_" for t in tokens)

def sha1_norm(text: str) -> str:
    norm = re.sub(r"\s+", " ", str(text).strip())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


# ==============================================================================
# 2. Jointly-Diagonalized Shared-Prompt Gaussian Fit & GPU L-BFGS Readout
# ==============================================================================
def _lw_shrinkage(E: np.ndarray) -> float:
    n, d = E.shape
    if n <= 1:
        return 1.0
    S = (E.T @ E) / n
    tr_S2 = np.sum(S ** 2)
    tr2_S = (np.trace(S) ** 2) / d
    num = np.sum((E @ E.T) ** 2) / (n ** 2) - tr_S2 / n
    den = tr_S2 - tr2_S
    return float(np.clip(num / max(den, 1e-12), 0.0, 1.0))

def fit_joint_diagonalized_gaussian(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    prompts_tr: np.ndarray,
    d_proj: int = 256,
    sp_scale: float = 1.0,
    iters: int = 4
) -> dict:
    """
    Fits the Shared-Prompt Gaussian in WPCA space and performs Cholesky-Eigen
    Joint Diagonalization (A Se A' = I, A Sp A' = diag(lam)) so that test features U
    and model centroids m can be passed directly into cavity_group_logpost(U, m, lam).
    """
    X = np.asarray(X_tr, dtype=np.float64)
    N, D = X.shape
    K = 18
    pu, pinv = np.unique(np.asarray(prompts_tr), return_inverse=True)
    G = len(pu)

    mx = X.mean(axis=0)
    sx = np.where(X.std(axis=0) > 1e-8, X.std(axis=0), 1.0)
    Xs = (X - mx) / sx

    # Within-Prompt PCA via Economy N x N Gram Matrix (since N << D)
    Pm = np.zeros((G, D), dtype=np.float64)
    np.add.at(Pm, pinv, Xs)
    Pm /= np.bincount(pinv)[:, None]
    R = Xs - Pm[pinv]

    d = int(min(d_proj, N - G - 1, D))
    if N < D:
        w_eig, U_eig = np.linalg.eigh(R @ R.T)
        ord_idx = np.argsort(w_eig)[::-1][:d]
        V = (R.T @ U_eig[:, ord_idx]) / np.sqrt(np.maximum(w_eig[ord_idx], 1e-12))[None, :]
    else:
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        V = Vt[:d].T

    Z = Xs @ V  # (N, d)

    # Two-Way Alternating Means: Z = mu[y] + P[prompt] + E
    P = np.zeros((G, d), dtype=np.float64)
    cnt_p = np.bincount(pinv, minlength=G).astype(np.float64)
    mu = np.zeros((K, d), dtype=np.float64)
    for _ in range(iters):
        res = Z - P[pinv]
        for c in range(K):
            mask_c = (y_tr == c)
            if mask_c.any():
                mu[c] = res[mask_c].mean(axis=0)
        res = Z - mu[y_tr]
        P.fill(0.0)
        np.add.at(P, pinv, res)
        P /= cnt_p[:, None]

    E = Z - mu[y_tr] - P[pinv]
    dof = max(N - G - K + 1, 1)
    Se_raw = (E.T @ E) / dof
    ae = _lw_shrinkage(E)
    Se = (1.0 - ae) * Se_raw + ae * (np.trace(Se_raw) / d) * np.eye(d)

    Pc = P - P.mean(axis=0, keepdims=True)
    Sp_raw = (Pc.T @ Pc) / max(G - 1, 1) - np.mean(1.0 / cnt_p) * Se_raw
    ap = _lw_shrinkage(Pc)
    tr_sp = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = ((1.0 - ap) * Sp_raw + ap * (tr_sp / d) * np.eye(d)) * sp_scale

    # Cholesky-Eigen Joint Diagonalization: A Se A' = I, A Sp A' = diag(lam)
    # Add numerical jitter to ensure strict positive-definiteness
    Se_jitter = Se + 1e-6 * np.eye(d)
    L = np.linalg.cholesky(Se_jitter)
    Li = np.linalg.inv(L)
    lam_raw, Q = np.linalg.eigh(Li @ Sp @ Li.T)
    order = np.argsort(lam_raw)[::-1]
    lam = np.maximum(lam_raw[order], 0.0)
    Q = Q[:, order]

    A = Q.T @ Li              # (d, d) whitening + rotation matrix
    W_proj = V @ A.T          # Combined (D, d) projection from Xs directly to whitened U
    m_whitened = mu @ A.T     # (18, d) whitened model centroids

    return {"mx": mx, "sx": sx, "W_proj": W_proj, "m": m_whitened, "lam": lam}

def transform_to_cavity_space(X: np.ndarray, gauss_prm: dict) -> np.ndarray:
    """Projects raw features X (n, D) directly into jointly-diagonalized space U (n, d)."""
    Xs = (np.asarray(X, dtype=np.float64) - gauss_prm["mx"]) / gauss_prm["sx"]
    return Xs @ gauss_prm["W_proj"]

def torch_lbfgs_lr(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, C: float = 1.0, iters: int = 200) -> np.ndarray:
    """Fast GPU/CPU L-BFGS Multinomial Logistic Regression."""
    if not HAS_TORCH:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=C, max_iter=iters, solver="lbfgs")
        lr.fit(X_tr, y_tr)
        return lr.predict_proba(X_te)

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    Xtr = torch.as_tensor(X_tr, dtype=torch.float32, device=dev)
    Xte = torch.as_tensor(X_te, dtype=torch.float32, device=dev)
    mu = Xtr.mean(dim=0)
    sd = Xtr.std(dim=0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    yt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)

    N, D = Xtr.shape
    W = torch.zeros((D, 18), dtype=torch.float32, device=dev, requires_grad=True)
    b = torch.zeros(18, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    reg = 1.0 / max(C * N, 1e-6)

    def closure():
        opt.zero_grad()
        logits = Xtr @ W + b
        loss = torch.nn.functional.cross_entropy(logits, yt) + 0.5 * reg * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits_te = Xte @ W + b
        probs_te = torch.softmax(logits_te, dim=-1).cpu().numpy()
    del Xtr, Xte, W, b
    return probs_te
