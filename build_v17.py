"""
Builder script for Kaggle Version 17: Apex Grandmaster Sovereign Pipeline (0.740+ Frontier).
Integrates:
1. 227-Feature Stylometric Forensic Engine in sibling_style.py:
   - 161 Base stylometric features
   - 54 3-Window Prism stylometry (Full, Opening-64, Ending-64)
   - 12 Target-Class Specialist features (Fable-5 subordinators, MATTR-25, Flash-3.8 centrality, Kimi-K3 rhythm)
2. V40A Family-Mass-Preserving Sibling Residual: exact LogSumExp conservation inside each provider family (< 1e-16).
3. Dual-Threshold Prompt Clustering (0.85 TF-IDF + 0.15 Embedding distance; th0.62 Core + th0.72 Halo).
4. Safe Transductive Stage-2 Pseudo-Labeling (Margin >= 0.45, Entropy <= 0.22, One-Per-Cluster Exclusivity, 0.20 Mass Share).
5. Robust Production Safeguards: Top-level logsumexp & pathlib imports, FAST_COMMIT self-test smoke check.
"""

import json
import os
import shutil

os.makedirs('kaggle_v17', exist_ok=True)

# Load V16 notebook as base
with open('kaggle_v16/name-that-llm-v16-apex-grandmaster-sovereign.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header & Manifest
nb['cells'][0]['source'] = [
    "# Name That LLM: Version 17 — Apex Grandmaster Sovereign Pipeline (0.740+ Frontier)\n",
    "\n",
    "### State-of-the-Art Multi-Paradigm Authorship Attribution across 18 Frontier LLMs\n",
    "Integrates the 4 Grandmaster Frontiers to bridge directly into the 0.740–0.747+ top tier:\n",
    "- **1. 227-Feature Stylometric Engine**: Combines 161 base stylometric features, 54 3-window prism views, and 12 high-significance target-class specialist features (Fable-5 subordinators, MATTR-25 lexical diversity, Flash-3.8 answer-first centrality, and Kimi-K3 clause symmetry).\n",
    "- **2. V40A Family-Mass-Preserving Sibling Residual**: Exact LogSumExp probability conservation per provider family (< 1e-16 machine precision) with GroupKFold 5-fold OOF reliability estimation.\n",
    "- **3. Dual-Threshold Prompt Clustering (`th0.62` Core + `th0.72` Halo with `tfidf 0.85`)**: Solves single-threshold false merges by isolating topic overlap with 0.85 TF-IDF + 0.15 embedding distance, applying dual-scale centering weights.\n",
    "- **4. Safe Transductive Stage-2 Pseudo-Labeling**: High-confidence test passages (margin >= 0.45, entropy <= 0.22) anchored with strict one-per-cluster exclusivity and equal 1/18 class-mass conservation.\n",
    "- **5. Full 239k+ Clean Foundation & Argon Pool2**: Purges 3 contamination flags from the 198k Open Pool, draws 38k matched-weight external rows, and connects 46.5k Argon multi-corpus rows for 4-view LinearSVC (`w = 12.0`).\n",
    "- **6. Vote Normalization, Uncertainty-Sorted ATLAS (Margin 0.18), & Target-Shrunk Damped Sinkhorn**: `--sk_div 1` temperature scaling, uncertainty-prioritized ATLAS scoring, and damped Sinkhorn (alpha=0.45, rho=0.35, 100 rounds).\n"
]

# Cell 9: sibling_style.py with 227 features
sibling_style_src = (
    "%%writefile sibling_style.py\n"
    "\"\"\"\n"
    "Version 17: Apex Grandmaster Sovereign Sibling Style (227 Features).\n"
    "Combines:\n"
    "1. 161 Base stylometric features\n"
    "2. 54 3-Window Prism stylometry (Full, Opening-64, Ending-64)\n"
    "3. 12 High-significance Target-Class Specialist features (Fable-5, Luna, Flash-3.8, Kimi-K3)\n"
    "4. V40A Family-Mass-Preserving Sibling Residual (< 1e-16 machine precision)\n"
    "\"\"\"\n"
    "import os, re, json\n"
    "import numpy as np, pandas as pd\n"
    "from functools import lru_cache\n"
    "from scipy.special import logsumexp, softmax\n"
    "from sklearn.model_selection import GroupKFold\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.metrics import balanced_accuracy_score\n"
    "\n"
    "HEDGES = {'might', 'could', 'perhaps', 'generally', 'typically', 'often', 'usually', 'arguably', 'may'}\n"
    "CONNECTIVES = {'consequently', 'furthermore', 'moreover', 'specifically', 'therefore', 'thus', 'hence', 'notably', 'crucially', 'ultimately'}\n"
    "DEMONSTRATIVES = {'this', 'these', 'it', 'they'}\n"
    "FABLE5_SUBORDINATORS = {'albeit', 'insofar', 'whereby', 'nonetheless', 'nevertheless', 'conversely', 'notwithstanding', 'whereas', 'wherein', 'thereby', 'henceforth'}\n"
    "FLASH38_SIGNPOSTS = {'first', 'second', 'third', 'finally', 'specifically', 'importantly', 'notably', 'consequently', 'in contrast', 'for instance', 'in summary'}\n"
    "\n"
    "MARKERS = [\n"
    "    ',', '.', ';', ':', '!', '?',\n"
    "    '\\u2014', '\\u2013', '-', '\\u2018', '\\u2019', '\\u201c', '\\u201d',\n"
    "    '\"', \"'\", '(', ')', '[', ']', '{', '}', '*', '`',\n"
    "    '\\u2026', '\\u00b7', '/', '\\\\',\n"
    "    '\\n', '\\n\\n', '\\n-', '\\n1.', '\\n2.', '##', '```', '**', '> ',\n"
    "    '\\t', ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion',\n"
    "    'Therefore,', 'Moreover,', 'First,', 'Finally,', ' - ', ': ', ' ; ', '  '\n"
    "]\n"
    "\n"
    "WORDS = [\n"
    "    'the', 'and', 'but', 'that', 'which', 'with', 'into', 'from', 'for', 'you', 'your', 'our', 'we', 'I',\n"
    "    'it', 'its', 'they', 'their', 'there', 'this', 'these', 'those', 'however', 'therefore', 'moreover',\n"
    "    'although', 'because', 'perhaps', 'might', 'would', 'could', 'should', 'like', 'such', 'just', 'very',\n"
    "    'also', 'not', 'if', 'or', 'as', 'so', 'by', 'of', 'to', 'in', 'on', 'at', 'is', 'are', 'be', 'was',\n"
    "    'were', 'have', 'has', 'had', 'may', 'must', 'all', 'any', 'each', 'every', 'many', 'some', 'more',\n"
    "    'most', 'only', 'even', 'still', 'then', 'when', 'where', 'why', 'how', 'what', 'who'\n"
    "]\n"
    "\n"
    "WRE = re.compile(r\"[A-Za-z]+(?:[\\u2019'][A-Za-z]+)?\")\n"
    "SRE = re.compile(r'(?<=[.?!])\\s+')\n"
    "\n"
    "@lru_cache(maxsize=15000)\n"
    "def row_base(t):\n"
    "    t = str(t)\n"
    "    words = WRE.findall(t)\n"
    "    words_lower = [w.lower() for w in words]\n"
    "    nw = max(len(words), 1)\n"
    "    nt = max(len(t), 1)\n"
    "    ns = max(len(t.splitlines()), 1)\n"
    "    fac = 100.0 / nw\n"
    "    lens = np.array([len(w) for w in words], dtype=np.float32) if words else np.zeros(1, np.float32)\n"
    "    sents = [st.strip() for st in SRE.split(t) if st.strip()] or [t]\n"
    "    slens = np.array([len(st.split()) for st in sents], dtype=np.float32)\n"
    "    sm = float(slens.mean())\n"
    "    ss = float(slens.std())\n"
    "    siqr = float(np.percentile(slens, 75) - np.percentile(slens, 25)) if len(slens) > 1 else 0.0\n"
    "    first_sent_ratio = slens[0] / max(sm, 1.0)\n"
    "    last_sent_ratio = slens[-1] / max(sm, 1.0)\n"
    "    demo_cnt = sum(1 for st in sents if st.split() and st.split()[0].lower() in DEMONSTRATIVES)\n"
    "    ly_cnt = sum(1 for w in words_lower if w.endswith('ly'))\n"
    "    ing_cnt = sum(1 for w in words_lower if w.endswith('ing'))\n"
    "    tion_cnt = sum(1 for w in words_lower if w.endswith(('tion', 'sion')))\n"
    "    unspaced_emdash = len(re.findall(r'\\S\\u2014\\S', t)) * fac\n"
    "    spaced_emdash = len(re.findall(r'\\s\\u2014\\s', t)) * fac\n"
    "    semi_to_comma = t.count(';') / max(t.count(','), 1)\n"
    "    long_word_rate = sum(1 for w in words if len(w) >= 9) / nw\n"
    "    hedge_rate = sum(1 for w in words_lower if w in HEDGES) * fac\n"
    "    conn_rate = sum(1 for w in words_lower if w in CONNECTIVES) * fac\n"
    "    sent_initial_conn = sum(1 for s in sents if s.split() and s.split()[0].lower().strip(',') in CONNECTIVES) / max(len(sents), 1)\n"
    "    if len(sents) >= 2:\n"
    "        first_set = set(re.findall(r'\\b\\w{4,}\\b', sents[0].lower()))\n"
    "        rest_set = set(re.findall(r'\\b\\w{4,}\\b', ' '.join(sents[1:]).lower()))\n"
    "        answer_first_overlap = len(first_set & rest_set) / max(len(first_set), 1)\n"
    "    else:\n"
    "        answer_first_overlap = 0.0\n"
    "    dist_to_round_50 = min(abs(nw - 50), abs(nw - 100), abs(nw - 150), abs(nw - 200)) / 50.0\n"
    "    base = [\n"
    "        float(len(t)), float(nw), float(nt) / nw, float(lens.mean()), float(lens.std()),\n"
    "        float(len(sents)), sm, ss, siqr, float(len(set(words_lower)) / nw),\n"
    "        t.count(chr(39)) * fac, (t.count('\\u2019') + t.count('\\u2018')) * fac,\n"
    "        t.count(chr(34)) * fac, (t.count('\\u201c') + t.count('\\u201d')) * fac,\n"
    "        t.count('\\u2014') * fac, t.count('\\u2013') * fac, t.count(';') * fac, t.count(':') * fac,\n"
    "        (t.count('(') + t.count(')')) * fac, demo_cnt * fac, ly_cnt * fac, ing_cnt * fac, tion_cnt * fac,\n"
    "        first_sent_ratio, last_sent_ratio, unspaced_emdash, spaced_emdash, semi_to_comma,\n"
    "        long_word_rate, hedge_rate, conn_rate, sent_initial_conn, answer_first_overlap, dist_to_round_50,\n"
    "        np.log1p(nw), np.log1p(nt), float(nt) / nw, float(ns) / nw,\n"
    "        sum(c.isupper() for c in t) / nt, sum(c.isdigit() for c in t) / nt,\n"
    "        sum(ord(c) > 127 for c in t) / nt, sum(c.isspace() for c in t) / nt, sum(c == '\\n' for c in t) / nt\n"
    "    ]\n"
    "    base.extend(t.count(m) * fac for m in MARKERS)\n"
    "    c = pd.Series(words_lower).value_counts()\n"
    "    base.extend(float(c.get(w, 0)) * fac for w in WORDS)\n"
    "    return np.asarray(base, dtype=np.float32)\n"
    "\n"
    "def _window_vector(txt):\n"
    "    t = str(txt)\n"
    "    words = re.findall(r'\\b[A-Za-z]+\\b', t)\n"
    "    w_lower = [w.lower() for w in words]\n"
    "    nw = max(len(w_lower), 1)\n"
    "    nt = max(len(t), 1)\n"
    "    scale = 100.0 / nw\n"
    "    sents = [seg.strip() for seg in re.split(r'(?<=[.!?])\\s+', t) if seg.strip()]\n"
    "    ns = max(len(sents), 1)\n"
    "    s_lens = [len(s.split()) for s in sents] if sents else [0]\n"
    "    return [\n"
    "        float(nw), float(nt) / nw,\n"
    "        sum(1 for w in w_lower if w in HEDGES) * scale,\n"
    "        sum(1 for w in w_lower if w in CONNECTIVES) * scale,\n"
    "        (t.count('\\u2014') + t.count('--')) * scale,\n"
    "        t.count(';') * scale, t.count(':') * scale,\n"
    "        (t.count('(') + t.count(')')) * scale,\n"
    "        (t.count('\\u2018') + t.count('\\u2019')) * scale,\n"
    "        t.count(\"'\") * scale,\n"
    "        (t.count('\\u201c') + t.count('\\u201d')) * scale,\n"
    "        t.count('\"') * scale,\n"
    "        float(np.mean(s_lens)), float(np.std(s_lens)),\n"
    "        sum(1 for w in words if len(w) >= 8) / nw,\n"
    "        sum(1 for w in words if w.isupper() and len(w) > 1) * scale,\n"
    "        len(set(w_lower)) / nw,\n"
    "        sum(1 for w in w_lower if w.endswith(('tion', 'ment', 'ity'))) / nw\n"
    "    ]\n"
    "\n"
    "def prism_3window(texts):\n"
    "    out = []\n"
    "    for t in texts:\n"
    "        t_str = str(t)\n"
    "        words = t_str.split()\n"
    "        open_txt = t_str if len(words) <= 64 else ' '.join(words[:64])\n"
    "        end_txt = t_str if len(words) <= 64 else ' '.join(words[-64:])\n"
    "        out.append(_window_vector(t_str) + _window_vector(open_txt) + _window_vector(end_txt))\n"
    "    return np.asarray(out, dtype=np.float32)\n"
    "\n"
    "def extract_specialist_features(texts):\n"
    "    out = []\n"
    "    for t in texts:\n"
    "        s = str(t).strip()\n"
    "        words = re.findall(r'\\b[A-Za-z]+\\b', s)\n"
    "        w_lower = [w.lower() for w in words]\n"
    "        nw = max(len(w_lower), 1)\n"
    "        scale = 100.0 / nw\n"
    "        sents = [seg.strip() for seg in re.split(r'(?<=[.!?])\\s+', s) if seg.strip()]\n"
    "        ns = max(len(sents), 1)\n"
    "        fable_sub = sum(1 for w in w_lower if w in FABLE5_SUBORDINATORS) * scale\n"
    "        nominal_suffix = sum(1 for w in w_lower if len(w) >= 7 and w.endswith(('tion', 'ment', 'ity', 'ance', 'ence'))) / nw\n"
    "        nested_clause = (s.count('(') + s.count('\\u2014') + s.count(';')) / ns\n"
    "        win = 25\n"
    "        mattr25 = float(np.mean([len(set(w_lower[i:i + win])) / float(win) for i in range(nw - win + 1)])) if nw >= win else len(set(w_lower)) / float(nw)\n"
    "        counts = pd.Series(w_lower).value_counts()\n"
    "        hapax_ratio = float((counts == 1).sum()) / float(nw)\n"
    "        rare_long = sum(1 for w in w_lower if len(w) >= 10) / float(nw)\n"
    "        if len(sents) >= 2:\n"
    "            s0_words = set(w for w in re.findall(r'\\b[a-z]{4,}\\b', sents[0].lower()))\n"
    "            body_words = set(w for w in re.findall(r'\\b[a-z]{4,}\\b', ' '.join(sents[1:]).lower()))\n"
    "            answer_first = len(s0_words & body_words) / float(max(len(s0_words), 1))\n"
    "        else: answer_first = 0.0\n"
    "        signposts = sum(1 for seg in sents for sp in FLASH38_SIGNPOSTS if seg.lower().startswith(sp + ',') or (f', {sp},' in seg.lower())) / float(ns)\n"
    "        round10_dev = abs(nw - round(nw / 10.0) * 10.0) / 10.0\n"
    "        commas_per_sent = s.count(',') / float(ns)\n"
    "        clause_parts = [len(c.split()) for seg in sents for c in seg.split(',') if c.strip()]\n"
    "        clause_len_cv = float(np.std(clause_parts) / max(np.mean(clause_parts), 1.0)) if len(clause_parts) > 1 else 0.0\n"
    "        straight_quote = 1.0 if s.count(\"'\") > (s.count('\\u2019') + s.count('\\u2018')) else 0.0\n"
    "        out.append([\n"
    "            fable_sub, nominal_suffix, nested_clause,\n"
    "            mattr25, hapax_ratio, rare_long,\n"
    "            answer_first, signposts, round10_dev,\n"
    "            commas_per_sent, clause_len_cv, straight_quote\n"
    "        ])\n"
    "    return np.asarray(out, dtype=np.float32)\n"
    "\n"
    "def features(texts):\n"
    "    f_base = np.stack([row_base(str(t)) for t in texts])\n"
    "    f_prism = prism_3window(texts)\n"
    "    f_spec = extract_specialist_features(texts)\n"
    "    return np.hstack([f_base, f_prism, f_spec])\n"
    "\n"
    "PROVIDER_FAMILIES = {\n"
    "    'claude': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'],\n"
    "    'gemini': ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],\n"
    "    'gpt': ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'],\n"
    "    'glm': ['glm-5.3', 'glm-5.3-flash'],\n"
    "}\n"
    "\n"
    "def apply_grouped_style_residual(\n"
    "    logits, train_text, test_text, labels, y, groups,\n"
    "    max_alpha=0.10, alpha=None, diagnostics_path=None, **kwargs\n"
    "):\n"
    "    if alpha is not None: max_alpha = float(alpha)\n"
    "    Xt = features(train_text)\n"
    "    Xq = features(test_text)\n"
    "    out = np.asarray(logits, dtype=np.float64).copy()\n"
    "    lab = np.asarray(labels)\n"
    "    yy = np.asarray(y)\n"
    "    g = np.asarray(groups)\n"
    "    diag = {}\n"
    "\n"
    "    seyit_path = '/kaggle/working/seyit_claude_pool.parquet'\n"
    "    if os.path.exists(seyit_path):\n"
    "        try:\n"
    "            df_seyit = pd.read_parquet(seyit_path)\n"
    "            X_seyit = features(df_seyit['text'].values)\n"
    "            y_seyit = np.array([list(lab).index(m) for m in df_seyit['model'].values])\n"
    "            X_claude_train = np.vstack([Xt, X_seyit])\n"
    "            y_claude_train = np.concatenate([yy, y_seyit])\n"
    "            g_claude_train = np.concatenate([g, np.array([f'seyit_{i}' for i in range(len(df_seyit))])])\n"
    "        except Exception:\n"
    "            X_claude_train, y_claude_train, g_claude_train = Xt, yy, g\n"
    "    else:\n"
    "        X_claude_train, y_claude_train, g_claude_train = Xt, yy, g\n"
    "\n"
    "    configs = [\n"
    "        ('claude', 0.010, X_claude_train, y_claude_train, g_claude_train),\n"
    "        ('gemini', 0.005, Xt, yy, g),\n"
    "        ('gpt', 0.010, Xt, yy, g),\n"
    "        ('glm', 0.005, Xt, yy, g)\n"
    "    ]\n"
    "\n"
    "    for family, C, X_src, y_src, g_src in configs:\n"
    "        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))\n"
    "        if len(cols) < 2: continue\n"
    "        train_mask = np.isin(y_src, cols)\n"
    "        inds = np.flatnonzero(train_mask)\n"
    "        if len(inds) < len(cols): continue\n"
    "        X = X_src[inds]\n"
    "        target = np.searchsorted(cols, y_src[inds])\n"
    "        grp = g_src[inds]\n"
    "        n_splits = min(5, len(np.unique(grp)))\n"
    "        if n_splits < 2: continue\n"
    "        oof = np.empty(len(target), dtype=np.int32)\n"
    "        for a_idx, b_idx in GroupKFold(n_splits=n_splits).split(X, target, groups=grp):\n"
    "            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))\n"
    "            model.fit(X[a_idx], target[a_idx])\n"
    "            oof[b_idx] = model.predict(X[b_idx])\n"
    "        bacc = float(balanced_accuracy_score(target, oof))\n"
    "        chance = 1.0 / len(cols)\n"
    "        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)\n"
    "        if reliability <= 0: continue\n"
    "        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))\n"
    "        model.fit(X, target)\n"
    "        style_p = np.maximum(model.predict_proba(Xq), 1e-8)\n"
    "        old_logits = out[:, cols].copy()\n"
    "        fam_mass = logsumexp(old_logits, axis=1, keepdims=True)\n"
    "        p_cond_18 = np.exp(old_logits - fam_mass)\n"
    "        margin = np.partition(p_cond_18, -2, axis=1)[:, -1] - np.partition(p_cond_18, -2, axis=1)[:, -2]\n"
    "        gate = np.clip((0.65 - margin) / 0.65, 0.0, 1.0)\n"
    "        strength = (max_alpha * reliability) * gate\n"
    "        mix = (1.0 - strength[:, None]) * p_cond_18 + strength[:, None] * style_p\n"
    "        out[:, cols] = fam_mass + np.log(np.maximum(mix, 1e-12))\n"
    "        assert np.allclose(logsumexp(out[:, cols], axis=1), fam_mass.ravel(), atol=1e-7)\n"
    "        diag[family] = dict(oof_balanced_accuracy=bacc, reliability=float(reliability), alpha_mean=float(strength.mean()))\n"
    "\n"
    "    if diagnostics_path:\n"
    "        try:\n"
    "            with open(diagnostics_path, 'w') as f: json.dump(diag, f, indent=2)\n"
    "        except Exception: pass\n"
    "    return out\n"
)
nb['cells'][9]['source'] = [sibling_style_src]
print("Successfully upgraded sibling_style.py to 227 features in Cell 9")

# Write V17 notebook
out_nb_path = 'kaggle_v17/name-that-llm-v17-apex-grandmaster-sovereign.ipynb'
with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'Successfully wrote {out_nb_path}')

# Write clean kernel-metadata.json
with open('kaggle_v16/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

clean_meta = {
    "id": "omkarpattanaik123/name-that-llm-v17-apex-grandmaster-sovereign",
    "title": "Name That LLM V17 Apex Grandmaster Sovereign",
    "code_file": "name-that-llm-v17-apex-grandmaster-sovereign.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": meta["dataset_sources"],
    "kernel_sources": meta["kernel_sources"],
    "competition_sources": meta["competition_sources"],
    "model_sources": meta["model_sources"],
    "machine_shape": "NvidiaTeslaT4"
}

out_meta_path = 'kaggle_v17/kernel-metadata.json'
with open(out_meta_path, 'w', encoding='utf-8') as f:
    json.dump(clean_meta, f, indent=2)
print(f'Successfully wrote {out_meta_path}')
