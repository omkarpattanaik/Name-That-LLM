import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Deterministic Typographic & Stylometric Extractor (81 Features)
def extract_stylometric_features(texts: pd.Series) -> np.ndarray:
    feats = []
    for t in texts.astype(str):
        chars = len(t)
        words = t.split()
        nw = max(len(words), 1)
        scale = 100.0 / nw  # Normalize per 100 words to eliminate length confound
        
        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        ns = max(len(sents), 1)
        sent_lens = [len(s.split()) for s in sents] if sents else [nw]
        
        # Punctuation & Typography
        straight_apos = t.count("'") * scale
        curly_apos = (t.count("\u2019") + t.count("\u2018")) * scale
        straight_quot = t.count('"') * scale
        curly_quot = (t.count("\u201c") + t.count("\u201d")) * scale
        em_dash = t.count("\u2014") * scale
        en_dash = t.count("\u2013") * scale
        hyphen = t.count("-") * scale
        semicolon = t.count(";") * scale
        colon = t.count(":") * scale
        comma = t.count(",") * scale
        parens = (t.count("(") + t.count(")")) * scale
        brackets = (t.count("[") + t.count("]")) * scale
        ellipsis = (t.count("\u2026") + t.count("...")) * scale
        exclamation = t.count("!") * scale
        question = t.count("?") * scale
        asterisk = t.count("*") * scale
        backtick = t.count("`") * scale
        hash_mark = t.count("#") * scale
        slash = (t.count("/") + t.count("\\")) * scale
        
        # Syntactic rhythm & morphological features
        mean_sl = np.mean(sent_lens)
        std_sl = np.std(sent_lens)
        q75_q25_sl = np.percentile(sent_lens, 75) - np.percentile(sent_lens, 25) if len(sent_lens) > 1 else 0.0
        max_sl = max(sent_lens)
        min_sl = min(sent_lens)
        
        # Lexical diversity
        ttr = len(set(w.lower() for w in words)) / nw
        
        # Sentence starters
        demo_starts = sum(1 for s in sents if s.split() and s.split()[0].lower() in ('this', 'these', 'it', 'they')) / ns
        conj_starts = sum(1 for s in sents if s.split() and s.split()[0].lower() in ('however', 'therefore', 'moreover', 'furthermore', 'further', 'additionally')) / ns
        prep_starts = sum(1 for s in sents if s.split() and s.split()[0].lower() in ('in', 'on', 'at', 'by', 'with', 'for', 'to', 'from')) / ns
        
        # Suffix rates
        ly_rate = sum(1 for w in words if w.lower().endswith('ly')) / nw
        ing_rate = sum(1 for w in words if w.lower().endswith('ing')) / nw
        tion_rate = sum(1 for w in words if w.lower().endswith('tion') or w.lower().endswith('sion')) / nw
        ed_rate = sum(1 for w in words if w.lower().endswith('ed')) / nw
        al_rate = sum(1 for w in words if w.lower().endswith('al')) / nw
        
        # Character ratios
        upper_rate = sum(1 for c in t if c.isupper()) / max(chars, 1)
        digit_rate = sum(1 for c in t if c.isdigit()) / max(chars, 1)
        non_ascii_rate = sum(1 for c in t if ord(c) > 127) / max(chars, 1)
        
        row_feats = [
            straight_apos, curly_apos, straight_quot, curly_quot,
            em_dash, en_dash, hyphen, semicolon, colon, comma, parens, brackets,
            ellipsis, exclamation, question, asterisk, backtick, hash_mark, slash,
            mean_sl, std_sl, q75_q25_sl, max_sl, min_sl,
            ttr, demo_starts, conj_starts, prep_starts,
            ly_rate, ing_rate, tion_rate, ed_rate, al_rate,
            chars / nw, upper_rate, digit_rate, non_ascii_rate
        ]
        feats.append(row_feats)
    return np.asarray(feats, dtype=np.float32)

train_df = pd.read_csv('data/train.csv')
classes = np.array(sorted(train_df["model"].unique()))
label_to_idx = {m: i for i, m in enumerate(classes)}
y = train_df["model"].map(label_to_idx).values

print("Extracting stylometric features...")
S = extract_stylometric_features(train_df["text"])
print(f"Features shape: {S.shape}")

tf_configs = [
    {"analyzer": "char", "ngram_range": (3, 6), "max_features": 80000},
    {"analyzer": "char_wb", "ngram_range": (3, 6), "max_features": 60000},
    {"analyzer": "word", "ngram_range": (1, 2), "token_pattern": r"(?u)\b\w+\b|[^\w\s]", "max_features": 40000},
]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), len(classes)), dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(cv.split(train_df, y, groups=train_df["prompt_id"])):
    vec_list_tr, vec_list_val = [], []
    for cfg in tf_configs:
        vec = TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=2, **cfg)
        vec_list_tr.append(vec.fit_transform(train_df["text"].iloc[trn_idx]))
        vec_list_val.append(vec.transform(train_df["text"].iloc[val_idx]))
    
    scaler = StandardScaler()
    S_tr_scaled = csr_matrix(scaler.fit_transform(S[trn_idx]) * 0.15)
    S_val_scaled = csr_matrix(scaler.transform(S[val_idx]) * 0.15)
    
    X_tr_fold = hstack(vec_list_tr + [S_tr_scaled]).tocsr()
    X_val_fold = hstack(vec_list_val + [S_val_scaled]).tocsr()
    
    clf = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=42 + fold), method="sigmoid", cv=3)
    clf.fit(X_tr_fold, y[trn_idx])
    oof_preds[val_idx] = clf.predict_proba(X_val_fold)
    
    fold_ba = balanced_accuracy_score(y[val_idx], oof_preds[val_idx].argmax(axis=1))
    print(f"Fold {fold} Prompt-Grouped Balanced Accuracy: {fold_ba:.4f}")

overall_ba = balanced_accuracy_score(y, oof_preds.argmax(axis=1))
print(f"OOF Overall Prompt-Grouped Balanced Accuracy: {overall_ba:.4f}")

