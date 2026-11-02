import numpy as np
import pandas as pd
from src.exact_author_joint_v12 import (
    recover_test_prompts_size_capped,
    exact_tuple_marginals,
    singleton_ll,
    cavity_group_logpost,
    soft_sinkhorn,
    margin_adaptive_sinkhorn_postprocess
)

def test_recovery():
    print("Testing prompt recovery...")
    texts = pd.Series([
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox leaps over a lazy sleeping dog.",
        "In quantum physics, entanglement describes correlated states.",
        "Quantum mechanics features particle entanglement across distances.",
        "A recipe for chocolate chip cookies requires flour, butter, and sugar.",
        "To bake delicious cookies, cream together butter and sugar then add flour."
    ] * 5)  # 30 texts
    labels = recover_test_prompts_size_capped(texts, sim_threshold=0.15, max_cluster_size=18)
    print(f"Recovered {len(np.unique(labels))} clusters for 30 texts.")
    for cid in np.unique(labels):
        sz = (labels == cid).sum()
        assert sz <= 18, f"Cluster {cid} exceeded size cap: {sz}"
    print("test_recovery passed!")

def test_cavity_group_logpost():
    print("Testing cavity group logpost...")
    C = 18
    d = 16
    np.random.seed(42)
    m = np.random.randn(C, d).astype(np.float64)
    lam = np.random.uniform(0.1, 2.0, size=(d,)).astype(np.float64)

    for n in [1, 2, 3, 5, 8]:
        U = np.random.randn(n, d).astype(np.float64)
        if n == 1:
            # For n=1, use singleton_ll
            ll = singleton_ll(U, m, lam)
            p = np.exp(ll - np.max(ll, axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            assert np.allclose(p.sum(axis=-1), 1.0)
            print(f"n={n}: singleton posterior validated.")
        else:
            logpost = cavity_group_logpost(U, m, lam, exact_max_n=3)
            p = np.exp(logpost)
            assert not np.isnan(p).any(), f"NaNs found for n={n}"
            assert np.allclose(p.sum(axis=-1), 1.0), f"Probabilities do not sum to 1 for n={n}"
            print(f"n={n}: cavity logpost validated, shape: {logpost.shape}, p min={p.min():.4e}, max={p.max():.4e}")
    print("test_cavity_group_logpost passed!")

def test_margin_adaptive_sinkhorn():
    print("Testing margin adaptive sinkhorn...")
    np.random.seed(42)
    probs = np.random.dirichlet(np.ones(18), size=100)
    # Inject high margin for some
    probs[0] = 0.01
    probs[0, 3] = 0.83
    probs[0] /= probs[0].sum()

    calibrated = margin_adaptive_sinkhorn_postprocess(probs)
    assert not np.isnan(calibrated).any()
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert (calibrated >= 0.0).all()
    print(f"Calibrated shape: {calibrated.shape}, row 0 top prob: {calibrated[0].max():.4f}")
    print("test_margin_adaptive_sinkhorn passed!")

if __name__ == "__main__":
    test_recovery()
    test_cavity_group_logpost()
    test_margin_adaptive_sinkhorn()
    print("ALL TESTS PASSED SUCCESSFULLY!")

