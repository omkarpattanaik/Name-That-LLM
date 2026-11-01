import pandas as pd
import numpy as np
import re
from test_v3_components import features
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
import sys

sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv('data/train.csv')
LABELS = sorted(train['model'].unique())
y = train['model'].map({m: i for i, m in enumerate(LABELS)}).values
groups = train['prompt_id'].values

print("Extracting features for all 2,699 train rows...")
X = features(train['text'].values)
print(f"Features extracted: {X.shape}")

lab = np.asarray(LABELS)

for family, C in [('claude', 0.003), ('gemini', 0.003), ('gpt', 0.003), ('glm', 0.03)]:
    cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
    train_mask = np.isin(y, cols)
    inds = np.flatnonzero(train_mask)
    X_fam = X[inds]
    target = np.searchsorted(cols, y[inds])
    grp = groups[inds]
    oof = np.empty(len(target), dtype=np.int32)
    
    for a, b in GroupKFold(n_splits=5).split(X_fam, target, groups=grp):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
        model.fit(X_fam[a], target[a])
        oof[b] = model.predict(X_fam[b])
        
    bacc = balanced_accuracy_score(target, oof)
    chance = 1.0 / len(cols)
    print(f"Family {family:8s} ({len(cols)} classes): OOF Balanced Accuracy = {bacc:.4f} (Random Chance: {chance:.4f}, Lift: +{bacc - chance:.4f})")

