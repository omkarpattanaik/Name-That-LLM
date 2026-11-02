import numpy as np
import pandas as pd
from scipy.special import softmax, log_softmax
from src.exact_author_joint_v12 import (
    recover_test_prompts_size_capped,
    soft_sinkhorn,
    margin_adaptive_sinkhorn_postprocess
)

def apply_exact_author_joint_postprocess(
    logits: np.ndarray,
    texts: list,
    sim_threshold: float = 0.26,
    max_cluster_size: int = 18,
    sinkhorn_strength: float = 0.78,
    base_mix: float = 0.64,
    low_margin_boost: float = 0.24,
    margin_scale: float = 0.35
) -> np.ndarray:
    """
    Complete Grandmaster Test-Time Bayesian Post-Processing:
    1. Size-capped UPGMA prompt clustering (max_cluster_size=18).
    2. Leave-one-out log-complement exclusivity repulsion inside clusters.
    3. Margin-Adaptive Confidence-Gated Soft Sinkhorn calibration.
    """
    n, C = logits.shape
    if n < 2:
        return logits

    p = softmax(logits, axis=1)

    # 1. Recover test prompt clusters
    labels = recover_test_prompts_size_capped(
        pd.Series(texts),
        sim_threshold=sim_threshold,
        max_cluster_size=max_cluster_size
    )

    # 2. Intra-cluster leave-one-out exclusivity repulsion
    p_repelled = p.copy()
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            continue
        cluster_p = p[idx]  # (k, C)
        # For each member, add log(1 - sum_{j!=i} q_j)
        for ii, i_idx in enumerate(idx):
            other_p = np.delete(cluster_p, ii, axis=0)  # (k-1, C)
            # Log complement penalty
            penalty = np.sum(np.log(np.clip(1.0 - other_p, 1e-4, 1.0)), axis=0)
            log_p_i = np.log(np.maximum(cluster_p[ii], 1e-12)) + 0.35 * penalty
            p_repelled[i_idx] = np.exp(log_p_i - log_p_i.max())
            p_repelled[i_idx] /= p_repelled[i_idx].sum()

    # 3. Margin-adaptive Sinkhorn calibration
    p_final = margin_adaptive_sinkhorn_postprocess(
        p_repelled,
        sinkhorn_strength=sinkhorn_strength,
        base_mix=base_mix,
        low_margin_boost=low_margin_boost,
        margin_scale=margin_scale
    )

    return np.log(np.maximum(p_final, 1e-12))

if __name__ == "__main__":
    print("Testing apply_exact_author_joint_postprocess...", flush=True)
    np.random.seed(42)
    N = 30
    C = 18
    logits = np.random.randn(N, C)
    texts = [f"Text passage number {i % 6} with specific content words" for i in range(N)]
    out_logits = apply_exact_author_joint_postprocess(logits, texts)
    print("Output logits shape:", out_logits.shape)
    p_out = softmax(out_logits, axis=1)
    assert np.allclose(p_out.sum(axis=1), 1.0)
    print("Test passed successfully!", flush=True)
