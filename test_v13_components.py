import sys
print("LINE 1: Starting test_v13_components.py...", flush=True)

import numpy as np
print("LINE 2: numpy imported", flush=True)

import pandas as pd
print("LINE 3: pandas imported", flush=True)

from src.argon_v13_apex_infer import (
    shape_text,
    small_words_text,
    sha1_norm,
    fit_joint_diagonalized_gaussian,
    transform_to_cavity_space,
    torch_lbfgs_lr
)
print("LINE 4: argon_v13_apex_infer imported", flush=True)

from src.exact_author_joint_v12 import cavity_group_logpost, margin_adaptive_sinkhorn_postprocess
print("LINE 5: exact_author_joint_v12 imported", flush=True)

def test_text_transforms():
    print("Testing text transforms...", flush=True)
    t = "Hello World! This is a test 123 with sophisticated terminology."
    st = shape_text(t)
    swt = small_words_text(t)
    print("Shape text:", st, flush=True)
    print("Small words text:", swt, flush=True)
    h = sha1_norm(t)
    print("SHA1 norm:", h, flush=True)
    assert len(h) == 16
    print("Text transforms passed!", flush=True)

def test_joint_diagonalization():
    print("Testing Joint Diagonalization...", flush=True)
    np.random.seed(42)
    N = 180  # 10 prompts x 18 models
    D = 128
    G = 10
    K = 18

    # Generate synthetic training features with prompt effect + model effect + noise
    prompts_tr = np.repeat(np.arange(G), K)
    y_tr = np.tile(np.arange(K), G)
    P_true = np.random.randn(G, D) * 1.5
    M_true = np.random.randn(K, D) * 1.0
    X_tr = P_true[prompts_tr] + M_true[y_tr] + np.random.randn(N, D) * 0.5

    prm = fit_joint_diagonalized_gaussian(X_tr, y_tr, prompts_tr, d_proj=32, iters=4)
    print("Fitted Gaussian keys:", list(prm.keys()), flush=True)
    print("W_proj shape:", prm["W_proj"].shape, flush=True)
    print("m shape:", prm["m"].shape, flush=True)
    print("lam shape:", prm["lam"].shape, flush=True)
    assert (prm["lam"] >= 0.0).all()

    # Project test passages
    N_te = 12
    X_te = np.random.randn(N_te, D)
    U = transform_to_cavity_space(X_te, prm)
    print("U shape:", U.shape, flush=True)
    assert U.shape == (N_te, prm["W_proj"].shape[1])

    # Pass into cavity_group_logpost
    logpost = cavity_group_logpost(U[:3], prm["m"], prm["lam"], exact_max_n=3)
    p = np.exp(logpost)
    print("Cavity logpost shape:", logpost.shape, flush=True)
    assert np.allclose(p.sum(axis=-1), 1.0)
    print("Joint Diagonalization + Cavity Logpost passed!", flush=True)

def test_torch_lr():
    print("Testing torch_lbfgs_lr...", flush=True)
    np.random.seed(42)
    N = 100
    D = 32
    X_tr = np.random.randn(N, D)
    y_tr = np.random.randint(0, 18, size=N)
    X_te = np.random.randn(15, D)

    probs = torch_lbfgs_lr(X_tr, y_tr, X_te, C=1.0, iters=50)
    print("Probs shape:", probs.shape, flush=True)
    assert probs.shape == (15, 18)
    assert np.allclose(probs.sum(axis=1), 1.0)
    print("torch_lbfgs_lr passed!", flush=True)

if __name__ == "__main__":
    test_text_transforms()
    test_joint_diagonalization()
    test_torch_lr()
    print("ALL V13 COMPONENTS VERIFIED AND PASSED LOCALLY!", flush=True)
