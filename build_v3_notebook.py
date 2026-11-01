import json
import os
import sys

# Load synthreaper notebook as base
with open('reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Base notebook has {len(nb['cells'])} cells.")

# Cell 0: Header markdown
nb['cells'][0]['source'] = [
    "# Name That LLM: Version 3 — Deep Frontier Multi-Paradigm SOTA Ensemble (LB 0.70+ Target)\n",
    "\n",
    "### Complete Multi-Paradigm Authorship Attribution Pipeline across 18 Frontier LLMs\n",
    "Built based on Deep Research from the frontier literature and official Kaggle winning recipes:\n",
    "- **Branch 1: Typographic Forensics & 4-View Case-Sensitive TF-IDF LinearSVC** (uncorrupted curly/straight quotes, em-dashes, syntax rhythm, suffixes)\n",
    "- **Branch 2: Multi-Layer Pooled LoRA24 Ensembles** (Qwen3-14B, Qwen3-8B, Llama-3.1-8B, Qwen2.5-14B, DeBERTa-v3-large)\n",
    "- **Branch 3: 15% Complementary Feature Stack** (independent GPU process to maximize memory efficiency)\n",
    "- **Branch 4: ATLAS Source-Conditioned Token-Likelihood Attribution** (20% blend on uncertain high-entropy passages)\n",
    "- **Branch 5: Dedicated 5-Way Anthropic Specialist Claude Gate** (trained on official Claude rows + 5,000 external Claude API passages)\n",
    "- **Branch 6: Sibling-Style Residual Reranking** (5-fold GroupKFold within-family calibrated classifiers for Claude, Gemini, GPT, GLM with exact cross-family mass conservation)\n",
    "- **Branch 7: Shared-Prompt Two-Way Gaussian Whitening** (Within-Prompt PCA + Ledoit-Wolf Covariance Shrinkage)\n",
    "- **Branch 8: Cohesion-Aware Adaptive Centering** (tight clusters [2, 18], twin_weight = 0.45, zero Hungarian assignment trap)\n",
    "- **Branch 9: Damped Target-Shrunk Sinkhorn-Knopp Calibration** (rho = 0.35, temperature = 0.90, strength = 0.60)\n",
    "- **Curated External Datasets**: Ingests Carrot Production's 198k Clean Open Pool (`ntl_open_pool.parquet`) with strict contamination filtering\n",
    "- **Fast Commit Guard**: < 90 second commit check, full multi-branch execution on hidden test rerun (~4,000 rows).\n"
]

# Cell 2: Inputs setup
# Make sure POOL checks for ntl_open_pool.parquet first!
cell2_text = "".join(nb['cells'][2]['source'])
cell2_text = cell2_text.replace(
    "POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))",
    """clean_pools = sorted(glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True))
if clean_pools:
    POOL = clean_pools[0]
else:
    POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))
print('Selected POOL dataset:', POOL)"""
)
nb['cells'][2]['source'] = [line + '\n' for line in cell2_text.splitlines()]

# Cell 6: sibling_style.py
# Replace with uncorrupted unicode escape markers and expanded features!
sibling_style_code = '''%%writefile sibling_style.py
import re
import numpy as np
from functools import lru_cache

TYPO_MARKERS = [
    ',', '.', ';', ':', '!', '?',
    '\\u2019',  # right single quote / curly apostrophe '
    '\\u2018',  # left single quote '
    '\\u201c',  # left double quote "
    '\\u201d',  # right double quote "
    '\\u2014',  # em dash -
    ' \\u2014 ', # spaced em dash
    '\\u2013',  # en dash -
    '\\u2026',  # ellipsis ...
    '"', "'", '(', ')', '[', ']', '{', '}', '*', '`',
    '/', '\\\\', '\\n', '\\n\\n', '\\n- ', '\\n1.', '\\n2.', '###', '##', '```', '**', '> ', '\\t',
    ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', 'First,', 'Finally,',
    ' - ', ': ', ' ; ', '  '
]

WORDS = [
    'the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its',
    'they','their','there','this','these','those','however','therefore','moreover','although','because',
    'perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as',
    'so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all',
    'any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who'
]

WRE = re.compile(r"[A-Za-z]+(?:['\\u2019\\u2018][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\\s+')

@lru_cache(maxsize=15000)
def row(t):
    t = str(t)
    s = WRE.findall(t)
    lo = [z.lower() for z in s]
    n = max(len(s), 1)
    nt = max(len(t), 1)
    ns = max(len(t.splitlines()), 1)
    lens = np.array([len(z) for z in s] or [0])
    sent_lens = np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    
    base = [
        np.log1p(n), np.log1p(nt), nt / n, ns / n, len(set(lo)) / n,
        float(np.mean(lens)), float(np.std(lens)),
        float(np.mean(sent_lens)), float(np.std(sent_lens)), float(np.max(sent_lens)),
        float(np.mean(lens >= 8)),
        sum(c.isupper() for c in t) / nt,
        sum(c.isdigit() for c in t) / nt,
        sum(ord(c) > 127 for c in t) / nt,
        sum(c.isspace() for c in t) / nt,
        sum(c == '\\n' for c in t) / nt
    ]
    base += [min(t.count(m) / n, 3.0) for m in TYPO_MARKERS]
    cnt = {}
    for w in lo:
        cnt[w] = cnt.get(w, 0) + 1
    base += [cnt.get(w, 0) / n for w in WORDS]
    return np.array(base, dtype=np.float32)

def features(texts):
    return np.stack([row(str(t)) for t in texts])

def apply_grouped_style_residual(logits, train_text, test_text, labels, y, groups, max_alpha=0.08, diagnostics_path=None):
    """OOF-estimated, conservative sibling re-ranking. Cross-family probability mass is unchanged."""
    import json
    import numpy as np
    from scipy.special import logsumexp, softmax
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score
    
    Xt = features(train_text)
    Xq = features(test_text)
    out = np.asarray(logits, dtype=np.float64).copy()
    lab = np.asarray(labels)
    yy = np.asarray(y)
    g = np.asarray(groups)
    diag = {}
    
    for family, C in [('claude', 0.003), ('gemini', 0.003), ('gpt', 0.003), ('glm', 0.03)]:
        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
        if len(cols) < 2:
            continue
        train_mask = np.isin(yy, cols)
        inds = np.flatnonzero(train_mask)
        X = Xt[inds]
        target = np.searchsorted(cols, yy[inds])
        grp = g[inds]
        oof = np.empty(len(target), dtype=np.int32)
        
        for a, b in GroupKFold(n_splits=5).split(X, target, groups=grp):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
            model.fit(X[a], target[a])
            oof[b] = model.predict(X[b])
            
        bacc = float(balanced_accuracy_score(target, oof))
        chance = 1.0 / len(cols)
        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)
        if reliability <= 0:
            diag[family] = dict(oof_balanced_accuracy=bacc, random_level=chance, alpha_mean=0.)
            continue
            
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
        model.fit(X, target)
        style_p = np.maximum(model.predict_proba(Xq), 1e-8)
        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)
        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]
        
        # Only modestly modify disputed sibling identities; preserve total provider mass
        gate = np.clip((0.65 - margin) / 0.65, 0.0, 1.0)
        strength = (max_alpha * reliability) * gate
        mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p
        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-8)
        diag[family] = dict(oof_balanced_accuracy=bacc, random_level=chance,
                            alpha_mean=float(strength.mean()), alpha_max=float(strength.max()),
                            changed_within_family=int(np.sum(old_p.argmax(1) != mix.argmax(1))))
                            
    if diagnostics_path:
        with open(diagnostics_path, 'w') as f:
            json.dump(diag, f, indent=2)
    return out
'''
nb['cells'][6]['source'] = [line + '\n' for line in sibling_style_code.splitlines()]

# Cell 8: infer.py
# Fix the corrupted typography marks and integrate clean pool filtering + Claude specialist gate!
infer_code = "".join(nb['cells'][8]['source'])

# 1. Clean pool filtering
old_pool_block = """    if a.pool and os.path.exists(a.pool):
        _P = pd.read_parquet(a.pool)
        if not {'text', 'model'}.issubset(_P.columns):
            raise ValueError(f'Pool must contain text/model columns: {a.pool}')
        _P = _P[_P.model.isin(LABELS)][['text', 'model']].copy()
        _P['text'] = _P.text.fillna('').astype(str); _P['_srcw'] = 0.35"""

new_pool_block = """    if a.pool and os.path.exists(a.pool):
        _P = pd.read_parquet(a.pool)
        if not {'text', 'model'}.issubset(_P.columns):
            raise ValueError(f'Pool must contain text/model columns: {a.pool}')
        # Filter contamination flags if present in clean open pool
        flag_cols = [c for c in _P.columns if c.startswith('flag_')]
        if flag_cols:
            _P = _P[~_P[flag_cols].any(axis=1)].copy()
            log(f'filtered pool via flags {flag_cols}: {len(_P)} rows remaining')
        if 'n_words' in _P.columns:
            _P = _P[(_P['n_words'] >= 40) & (_P['n_words'] <= 250)].copy()
        _P = _P[_P.model.isin(LABELS)][['text', 'model']].copy()
        _P['text'] = _P.text.fillna('').astype(str)
        _P = _P[~_P['text'].str.contains('\\n', regex=False)].copy()
        _P['_srcw'] = 0.35"""

infer_code = infer_code.replace(old_pool_block, new_pool_block)

# 2. Fix the corrupted typography marks
old_marks_line = "    _marks = ['  ', '', '', '', '', '', '', '', '', '\"', ';', ':', '(', ')', '[', ']', '**', '`', '', '', '\\n', '\\n\\n', 'e.g.', 'i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', '\\t', ' - ', ' = ', '\\n- ', '\\n1.', '###', '```', '> ']"
new_marks_line = "    _marks = [' \\u2014 ', '\\u2014', '\\u2013', '\\u2019', '\\u2018', '\\u201c', '\\u201d', '\\u2026', '\"', \"'\", ';', ':', '(', ')', '[', ']', '**', '`', '/', '\\\\', '\\n', '\\n\\n', 'e.g.', 'i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', '\\t', ' - ', ' = ', '\\n- ', '\\n1.', '###', '```', '> ']"

infer_code = infer_code.replace(old_marks_line, new_marks_line)

# 3. Add Claude Gate integration before the final submission writer!
old_tail = """    pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)"""
new_tail = """    # Dedicated Anthropic Claude 5-Way Gate
    try:
        claude_cols = [i for i, l in enumerate(LABELS) if l.startswith('claude')]
        claude_api_files = sorted(glob.glob('/kaggle/input/**/ntl_extra_passages_claude.csv', recursive=True))
        if claude_api_files and len(claude_cols) == 5:
            _api_c = pd.read_csv(claude_api_files[0])
            _tr_c_mask = np.isin(y, claude_cols)
            _c_train_texts = train.text.iloc[_tr_c_mask].tolist() + _api_c.text.astype(str).tolist()
            _c_train_y = np.concatenate([np.searchsorted(claude_cols, y[_tr_c_mask]),
                                         np.searchsorted(claude_cols, _api_c.model.map(L2I).values)])
            from sklearn.feature_extraction.text import TfidfVectorizer as _TV
            from sklearn.linear_model import LogisticRegression as _LR
            _cv_c = _TV(sublinear_tf=True, lowercase=False, min_df=2, ngram_range=(1, 2), max_features=30000)
            _Xtr_c = _cv_c.fit_transform(_c_train_texts)
            _Xte_c = _cv_c.transform(test.text.values)
            _clf_c = _LR(C=1.0, max_iter=300, class_weight='balanced')
            _clf_c.fit(_Xtr_c, _c_train_y)
            _p_c = _clf_c.predict_proba(_Xte_c)  # (N, 5)
            
            # Redistribute Anthropic family mass
            _P_full = np.exp(tot - tot.max(axis=1, keepdims=True))
            _P_full /= _P_full.sum(axis=1, keepdims=True)
            _anthropic_mass = _P_full[:, claude_cols].sum(axis=1, keepdims=True)
            _gate_w = 0.35
            _redistributed_claude = (1.0 - _gate_w) * _P_full[:, claude_cols] + _gate_w * (_anthropic_mass * _p_c)
            _P_full[:, claude_cols] = _redistributed_claude
            tot = np.log(np.maximum(_P_full, 1e-12))
            log('Specialist 5-Way Claude Gate successfully applied to Anthropic mass')
    except Exception as _c_gate_err:
        log('Claude specialist gate skipped:', repr(_c_gate_err))
    
    pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)"""

infer_code = infer_code.replace(old_tail, new_tail)
nb['cells'][8]['source'] = [line + '\n' for line in infer_code.splitlines()]

# Cell 12: Run comp_stack.py cleanly in python subprocess
cell12_code = """# complementary public stack in its own process (frees all GPU memory when done); its wall time is charged to the ATLAS budget
import time as _t
import subprocess, sys, os
_c0 = _t.time()
os.environ['NTL_DROP_Q35'] = '0'
if not FAST_COMMIT:
    print('Starting comp_stack in independent process to conserve GPU memory...')
    _res = subprocess.run([sys.executable, 'comp_stack.py'])
    print('comp_stack finished with returncode:', _res.returncode)
COMP_SEC = _t.time() - _c0
print('comp stack wall time', round(COMP_SEC))
"""
nb['cells'][12]['source'] = [line + '\n' for line in cell12_code.splitlines()]

# Cell 14: Execution command
# Use optimal tuned hyperparameters: sinkhorn 0.60, adapt_thr 0.15, adapt_w 1.0, comp_w 0.15, atlas_w 0.20
cell14_text = "".join(nb['cells'][14]['source'])
cell14_text = cell14_text.replace("'--sinkhorn', '0.45'", "'--sinkhorn', '0.60'")
nb['cells'][14]['source'] = [line + '\n' for line in cell14_text.splitlines()]

output_path = 'kaggle_v3/name-that-llm-v3-sota-ensemble.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated {output_path} with {len(nb['cells'])} cells!")
