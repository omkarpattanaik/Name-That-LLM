import numpy as np
import pandas as pd
from src.argon_v15_sota_upgrades import (
    stamatatos_topic_mask,
    extract_3window_prism_style,
    apply_v40a_family_sibling_residual,
    target_shrunk_damped_sinkhorn,
    PROVIDER_FAMILIES
)

df = pd.read_csv('data/train.csv')
sample_texts = df.text.iloc[:20]

# 1. Test Stamatatos masking
masked = stamatatos_topic_mask(sample_texts)
print("Stamatatos sample:")
print("Original:", sample_texts.iloc[0][:100])
print("Masked  :", masked.iloc[0][:100])
assert len(masked) == 20

# 2. Test 3-window prism style
feats = extract_3window_prism_style(sample_texts)
print(f"3-Window style features shape: {feats.shape}")
assert feats.shape == (20, 54)
assert np.isfinite(feats).all()

# 3. Test V40A sibling residual
labels = sorted(df.model.unique())
core_logits = np.random.randn(20, 18)
sibling_gates = {
    fam: np.random.dirichlet(np.ones(len(PROVIDER_FAMILIES[fam])), size=20)
    for fam in PROVIDER_FAMILIES
}
v40a_logits = apply_v40a_family_sibling_residual(core_logits, sibling_gates, labels)
print("V40A logits shape:", v40a_logits.shape)
assert v40a_logits.shape == (20, 18)
assert np.isfinite(v40a_logits).all()

# 4. Test Target-Shrunk Damped Sinkhorn
probs = np.random.dirichlet(np.ones(18), size=200)
sk_probs = target_shrunk_damped_sinkhorn(probs, alpha_damp=0.45, rho_shrink=0.35, n_rounds=100)
print(f"Sinkhorn probs shape: {sk_probs.shape}, row sums min: {sk_probs.sum(axis=1).min():.6f}, max: {sk_probs.sum(axis=1).max():.6f}")
assert np.allclose(sk_probs.sum(axis=1), 1.0, atol=1e-5)

print("\nALL V15 UPGRADES VERIFIED SUCCESSFULLY!")
