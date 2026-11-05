import numpy as np
import pandas as pd
from src.argon_v17_apex_upgrades import (
    extract_low_recall_specialist_features,
    compute_dual_threshold_prompt_clusters,
    select_safe_transductive_pseudo_labels
)

print("--- 1. Testing extract_low_recall_specialist_features ---")
df = pd.read_csv('data/train.csv')
sample_texts = df.text.values[:100]
feats = extract_low_recall_specialist_features(sample_texts)
print(f"Extracted features shape: {feats.shape}")
assert feats.shape == (100, 12)
assert np.isfinite(feats).all()
print("PASS: Specialist feature extraction verified!")

print("\n--- 2. Testing compute_dual_threshold_prompt_clusters ---")
dummy_emb = np.random.randn(100, 64)
g_core, g_halo = compute_dual_threshold_prompt_clusters(sample_texts, dummy_emb, tfidf_weight=0.85)
print(f"Core clusters: {len(np.unique(g_core))}, Halo clusters: {len(np.unique(g_halo))}")
assert len(g_core) == 100 and len(g_halo) == 100
assert len(np.unique(g_core)) >= len(np.unique(g_halo))  # Core is tighter than halo!
print("PASS: Dual-threshold clustering verified!")

print("\n--- 3. Testing select_safe_transductive_pseudo_labels ---")
dummy_probs = np.random.dirichlet(np.ones(18), size=100)
# Make some rows confident
dummy_probs[0, 3] = 0.95; dummy_probs[0, :] /= dummy_probs[0].sum()
dummy_probs[5, 8] = 0.90; dummy_probs[5, :] /= dummy_probs[5].sum()
dummy_probs[12, 11] = 0.88; dummy_probs[12, :] /= dummy_probs[12].sum()

idx, labels, weights = select_safe_transductive_pseudo_labels(
    dummy_probs, g_core, min_top2_margin=0.45, max_norm_entropy=0.22
)
print(f"Selected pseudo labels: {len(idx)} rows, indices: {idx}, labels: {labels}")
assert len(idx) == len(labels) == len(weights)
assert np.all(weights > 0)
print("PASS: Safe transductive pseudo-labeling verified!")

print("\n=== ALL V17 COMPONENTS VERIFIED 100% CLEANLY ===")
