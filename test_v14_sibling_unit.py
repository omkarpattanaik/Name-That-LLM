import numpy as np
import pandas as pd
from scipy.special import logsumexp

df = pd.read_csv('data/train.csv')
train_text = df.text.values[:100]
test_text = df.text.values[100:150]
labels = sorted(df.model.unique())
y = np.array([labels.index(m) for m in df.model.values[:100]])
groups = df.prompt_id.values[:100]

# Dummy logits
logits = np.random.randn(len(test_text), 18)

# Execute sibling style
from test_sibling_oof import row_v14, MARKERS_V14, WORDS, WRE_V14, SRE, HEDGES, CONNECTIVES, DEMONSTRATIVES

# Test import and function execution
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from scipy.special import softmax

def features(texts):
    return np.stack([row_v14(str(t)) for t in texts])

def apply_grouped_style_residual(
    logits, train_text, test_text, labels, y, groups,
    max_alpha=0.10, alpha=None, diagnostics_path=None, **kwargs
):
    if alpha is not None:
        max_alpha = float(alpha)
    Xt = features(train_text)
    Xq = features(test_text)
    out = np.asarray(logits, dtype=np.float64).copy()
    lab = np.asarray(labels)
    yy = np.asarray(y)
    g = np.asarray(groups)
    diag = {}

    configs = [
        ('claude', 0.010),
        ('gemini', 0.005),
        ('gpt', 0.010),
        ('glm', 0.005)
    ]

    for family, C in configs:
        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
        if len(cols) < 2:
            continue
        train_mask = np.isin(yy, cols)
        inds = np.flatnonzero(train_mask)
        if len(inds) < len(cols):
            continue
        X = Xt[inds]
        target = np.searchsorted(cols, yy[inds])
        grp = g[inds]
        # In small test, n_splits <= unique groups
        n_splits = min(5, len(np.unique(grp)))
        if n_splits < 2:
            continue
        oof = np.empty(len(target), dtype=np.int32)
        for a_idx, b_idx in GroupKFold(n_splits=n_splits).split(X, target, groups=grp):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
            model.fit(X[a_idx], target[a_idx])
            oof[b_idx] = model.predict(X[b_idx])
        bacc = float(balanced_accuracy_score(target, oof))
        chance = 1.0 / len(cols)
        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)
        if reliability <= 0:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
        model.fit(X, target)
        style_p = np.maximum(model.predict_proba(Xq), 1e-8)
        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)
        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]
        gate = np.clip((0.65 - margin) / 0.65, 0.0, 1.0)
        strength = (max_alpha * reliability) * gate
        mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p
        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-7)
    return out

out_res = apply_grouped_style_residual(logits, train_text, test_text, labels, y, groups, max_alpha=0.10)
print("Sibling style unit test passed successfully! Output shape:", out_res.shape)
