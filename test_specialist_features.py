import time, re
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

df = pd.read_csv('data/train.csv')
t0 = time.time()

FABLE5_SUBORDINATORS = {
    'albeit', 'insofar', 'whereby', 'nonetheless', 'nevertheless', 'conversely',
    'notwithstanding', 'whereas', 'wherein', 'thereby', 'henceforth'
}
FLASH38_SIGNPOSTS = {
    'first', 'second', 'third', 'finally', 'specifically', 'importantly',
    'notably', 'consequently', 'in contrast', 'for instance', 'in summary'
}

def extract_features(texts):
    out = []
    for s in texts:
        s = str(s).strip()
        words = re.findall(r'\b[A-Za-z]+\b', s)
        w_lower = [w.lower() for w in words]
        nw = max(len(w_lower), 1)
        scale = 100.0 / nw

        sents = [seg.strip() for seg in re.split(r'(?<=[.!?])\s+', s) if seg.strip()]
        ns = max(len(sents), 1)

        # Fable-5 features
        fable_sub = sum(1 for w in w_lower if w in FABLE5_SUBORDINATORS) * scale
        nominal_suffix = sum(1 for w in w_lower if len(w) >= 7 and w.endswith(('tion', 'ment', 'ity', 'ance', 'ence'))) / nw
        nested_clause = (s.count('(') + s.count('\u2014') + s.count(';')) / ns

        # Luna features (MATTR-25 & Hapax)
        win = 25
        if nw >= win:
            mattr25 = float(np.mean([len(set(w_lower[i:i + win])) / float(win) for i in range(nw - win + 1)]))
        else:
            mattr25 = len(set(w_lower)) / float(nw)
        counts = pd.Series(w_lower).value_counts()
        hapax_ratio = float((counts == 1).sum()) / float(nw)
        rare_long = sum(1 for w in w_lower if len(w) >= 10) / float(nw)

        # Flash-3.8 features
        if len(sents) >= 2:
            s0_words = set(w for w in re.findall(r'\b[a-z]{4,}\b', sents[0].lower()))
            body_words = set(w for w in re.findall(r'\b[a-z]{4,}\b', ' '.join(sents[1:]).lower()))
            answer_first = len(s0_words & body_words) / float(max(len(s0_words), 1))
        else:
            answer_first = 0.0

        signposts = sum(1 for seg in sents for sp in FLASH38_SIGNPOSTS if seg.lower().startswith(sp + ',') or (f', {sp},' in seg.lower())) / float(ns)
        round10_dev = abs(nw - round(nw / 10.0) * 10.0) / 10.0

        # Kimi-K3 features
        commas_per_sent = s.count(',') / float(ns)
        clause_parts = [len(c.split()) for seg in sents for c in seg.split(',') if c.strip()]
        clause_len_cv = float(np.std(clause_parts) / max(np.mean(clause_parts), 1.0)) if len(clause_parts) > 1 else 0.0
        straight_quote = 1.0 if s.count("'") > (s.count('\u2019') + s.count('\u2018')) else 0.0

        out.append([
            fable_sub, nominal_suffix, nested_clause,
            mattr25, hapax_ratio, rare_long,
            answer_first, signposts, round10_dev,
            commas_per_sent, clause_len_cv, straight_quote
        ])
    return np.asarray(out, dtype=np.float32)

X_spec = extract_features(df['text'].values)
elapsed = time.time() - t0
print(f"Extracted {X_spec.shape[1]} features across {len(X_spec)} rows in {elapsed:.2f}s")

feature_names = ['fable_sub', 'nominal_suffix', 'nested_clause', 'mattr25', 'hapax_ratio', 'rare_long', 'answer_first', 'signposts', 'round10_dev', 'commas_per_sent', 'clause_len_cv', 'straight_quote']

# Test on Claude family
claude_mask = df.model.str.startswith('claude').values
y_claude = df.loc[claude_mask, 'model'].astype('category').cat.codes.values
F_scores, p_vals = f_classif(X_spec[claude_mask], y_claude)
print("\n--- CLAUDE FAMILY STATISTICAL DISCRIMINATION ---")
for name, f, p in zip(feature_names, F_scores, p_vals):
    if p < 0.01:
        print(f"  {name:20s} F={f:6.2f} (p={p:.2e})")

# Test on GPT family
gpt_mask = df.model.str.startswith('gpt').values
y_gpt = df.loc[gpt_mask, 'model'].astype('category').cat.codes.values
F_scores, p_vals = f_classif(X_spec[gpt_mask], y_gpt)
print("\n--- GPT FAMILY STATISTICAL DISCRIMINATION ---")
for name, f, p in zip(feature_names, F_scores, p_vals):
    if p < 0.01:
        print(f"  {name:20s} F={f:6.2f} (p={p:.2e})")

# Test on Gemini family
gem_mask = df.model.str.startswith('gemini').values
y_gem = df.loc[gem_mask, 'model'].astype('category').cat.codes.values
F_scores, p_vals = f_classif(X_spec[gem_mask], y_gem)
print("\n--- GEMINI FAMILY STATISTICAL DISCRIMINATION ---")
for name, f, p in zip(feature_names, F_scores, p_vals):
    if p < 0.01:
        print(f"  {name:20s} F={f:6.2f} (p={p:.2e})")

