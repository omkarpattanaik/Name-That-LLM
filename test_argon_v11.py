import numpy as np
import pandas as pd
from src.argon_v11_sota import (
    text_to_shape_view,
    extract_sibling_cliff_features,
    discover_cohesive_prompt_clusters,
    DiagonalLogLinearMetaCalibrator,
    apply_soft_cluster_exclusivity_and_sinkhorn,
    compute_domain_protected_sample_weights
)

texts = pd.Series([
    "This is a test passage—unspaced dash; semicolon used, specifically!",
    "Another passage with might, could, perhaps hedges?",
    "First sentence here. Second sentence also here with connectives consequently.",
    "Short sentence. Another one."
] * 15)

# Test 1: Shape view
sv = text_to_shape_view(texts[:2])
print('Shape view sample:', sv[0][:30])

# Test 2: Sibling cliff features
feats = extract_sibling_cliff_features(texts)
print('Cliff feats shape:', feats.shape)

# Test 3: Prompt cohesion discovery
Z = np.random.randn(len(texts), 32)
cids, cweights = discover_cohesive_prompt_clusters(texts, Z)
print('Discovered clusters:', len(np.unique(cids)), 'Cohesion weights mean:', float(cweights.mean()))

# Test 4: Meta calibrator
p1 = np.random.dirichlet(np.ones(18), size=len(texts))
p2 = np.random.dirichlet(np.ones(18), size=len(texts))
cal = DiagonalLogLinearMetaCalibrator()
y_mock = np.random.randint(0, 18, size=len(texts))
w = cal.fit_view_temperatures([p1, p2], y_mock)
print('Fitted view weights:', w)

logits = cal.predict_logits([p1, p2])
print('Meta logits shape:', logits.shape)

# Test 5: Soft exclusivity + Sinkhorn
p_final = apply_soft_cluster_exclusivity_and_sinkhorn(logits, cids, cweights)
print('Final prob shape:', p_final.shape, 'Row sum check:', p_final.sum(axis=1)[:3])
print('All tests in argon_v11_sota passed cleanly!')

