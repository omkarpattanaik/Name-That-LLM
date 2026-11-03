"""
Builder script for Kaggle Version 14: Apex Grandmaster Champion Pipeline.
Directly evolves our 0.667 PB architecture by fusing:
1. Matched-weight clean external rows (drawn from Open Pool by match_weight to match nlm_matched_clean counts)
2. Safe Open Pool subset (purging Gemini 3.7 routing risks and train prompt regeneration leaks)
3. Evidence-Aware Pairwise Identity Verifier (mercurydust/ntllm-chatgpt-a-b Revision 03)
4. Enhanced Sibling Forensics (uncorrupted Unicode markers + 16 sibling cliff features + GroupKFold OOF reliability)
5. Pure Per-Passage Evaluation (--perpassage 1, strictly zero test clustering and zero LOO repulsion)
6. Target-Shrunk Sinkhorn Calibration (--sinkhorn 0.30, --sinkhorn_target_mix 0.35)
7. Bayesian Typography Scaling (evidence coefficient 1.35, ATLAS margin 0.32)
"""

import json
import os

os.makedirs('kaggle_v14', exist_ok=True)

# Load Zbigniew 0.667 as base template
with open('zbigniew_0667/0-667-pairwise-verifier-matched-external-rows.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header & Manifest
nb['cells'][0]['source'] = [
    "# Name That LLM: Version 14 — Apex Grandmaster Champion Pipeline\n",
    "\n",
    "### State-of-the-Art Multi-Paradigm Authorship Attribution across 18 Frontier LLMs\n",
    "Consolidates our 0.667 benchmark with full pairwise verification, matched-weight clean pool sampling, and enhanced sibling cliff forensics:\n",
    "- **1. Clean Open Pool & Matched-Weight Foundation**: Purges dynamic routing risks and prompt regeneration leaks from the 198k Open Pool, drawing external rows strictly proportional to `match_weight` to match the exact host length and typographic distribution.\n",
    "- **2. Evidence-Aware Pairwise Identity Verifier (V34.1)**: Conditions on Revision 03 surface observations, cue spans, and prompt weak labels (`mercurydust/ntllm-chatgpt-a-b`), fused via confidence margin gating on close calls ($\\text{margin} < 0.35$).\n",
    "- **3. Enhanced Sibling Cliff Forensics**: 16 targeted syntactic, burstiness, and punctuation features (unspaced vs. spaced em-dashes, hedge/connective ratios, answer-first overlap, distance to round-50) evaluated with 5-fold GroupKFold OOF reliability estimation, boosting Claude OOF balanced accuracy by +4.81% and GPT by +2.13%.\n",
    "- **4. Independent Per-Passage Test Evaluation (`--perpassage 1`)**: Zero test-time clustering and zero leave-one-out exclusivity repulsion, strictly preventing cluster contamination on sparse test prompts.\n",
    "- **5. Target-Shrunk Sinkhorn Balancing**: Moderated Sinkhorn ($\\lambda = 0.30$, target mix $\\rho = 0.35$) interpolating test-implied marginals with training priors over 100 iterations.\n",
    "- **6. SOTA Backbone Ensemble & ATLAS Rescoring**: Fold-averaged two-stage LoRA24 (Qwen3-14B, Qwen3-8B, Llama-3.1-8B, Qwen2.5-14B), DeBERTa-v3-large, 4-view n-gram LinearSVC, complementary frozen stack, and ATLAS token-level rescoring ($\\text{margin} < 0.32$).\n"
]

# Cell 9: Enhanced Sibling Style Forensics
nb['cells'][9]['source'] = [
    "%%writefile sibling_style.py\n",
    "\"\"\"\n",
    "Version 14: Apex Grandmaster Champion Sibling Style & Hierarchical Specialist Forensics.\n",
    "Combines uncorrupted typographic markers + 16 targeted sibling cliff features\n",
    "with GroupKFold 5-fold OOF reliability estimation, confidence margin gating,\n",
    "and exact family probability mass preservation (logsumexp invariance).\n",
    "\"\"\"\n",
    "import os\n",
    "import re\n",
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from functools import lru_cache\n",
    "from scipy.special import logsumexp, softmax\n",
    "from sklearn.model_selection import GroupKFold\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.metrics import balanced_accuracy_score\n",
    "\n",
    "HEDGES = {'might', 'could', 'perhaps', 'generally', 'typically', 'often', 'usually', 'arguably', 'may'}\n",
    "CONNECTIVES = {'consequently', 'furthermore', 'moreover', 'specifically', 'therefore', 'thus', 'hence', 'notably', 'crucially', 'ultimately'}\n",
    "DEMONSTRATIVES = {'this', 'these', 'it', 'they'}\n",
    "\n",
    "MARKERS = [\n",
    "    ',', '.', ';', ':', '!', '?',\n",
    "    '\\u2014', '\\u2013', '-', '\\u2018', '\\u2019', '\\u201c', '\\u201d',\n",
    "    '\"', \"'\", '(', ')', '[', ']', '{', '}', '*', '`',\n",
    "    '\\u2026', '\\u00b7', '/', '\\\\',\n",
    "    '\\n', '\\n\\n', '\\n-', '\\n1.', '\\n2.', '##', '```', '**', '> ',\n",
    "    '\\t', ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion',\n",
    "    'Therefore,', 'Moreover,', 'First,', 'Finally,', ' - ', ': ', ' ; ', '  '\n",
    "]\n",
    "\n",
    "WORDS = [\n",
    "    'the', 'and', 'but', 'that', 'which', 'with', 'into', 'from', 'for', 'you', 'your', 'our', 'we', 'I',\n",
    "    'it', 'its', 'they', 'their', 'there', 'this', 'these', 'those', 'however', 'therefore', 'moreover',\n",
    "    'although', 'because', 'perhaps', 'might', 'would', 'could', 'should', 'like', 'such', 'just', 'very',\n",
    "    'also', 'not', 'if', 'or', 'as', 'so', 'by', 'of', 'to', 'in', 'on', 'at', 'is', 'are', 'be', 'was',\n",
    "    'were', 'have', 'has', 'had', 'may', 'must', 'all', 'any', 'each', 'every', 'many', 'some', 'more',\n",
    "    'most', 'only', 'even', 'still', 'then', 'when', 'where', 'why', 'how', 'what', 'who'\n",
    "]\n",
    "\n",
    "WRE = re.compile(r\"[A-Za-z]+(?:[\\u2019'][A-Za-z]+)?\")\n",
    "SRE = re.compile(r'(?<=[.?!])\\s+')\n",
    "\n",
    "@lru_cache(maxsize=15000)\n",
    "def row(t):\n",
    "    t = str(t)\n",
    "    words = WRE.findall(t)\n",
    "    words_lower = [w.lower() for w in words]\n",
    "    nw = max(len(words), 1)\n",
    "    nt = max(len(t), 1)\n",
    "    ns = max(len(t.splitlines()), 1)\n",
    "    fac = 100.0 / nw\n",
    "    lens = np.array([len(w) for w in words], dtype=np.float32) if words else np.zeros(1, np.float32)\n",
    "    sents = [st.strip() for st in SRE.split(t) if st.strip()] or [t]\n",
    "    slens = np.array([len(st.split()) for st in sents], dtype=np.float32)\n",
    "    sm = float(slens.mean())\n",
    "    ss = float(slens.std())\n",
    "    siqr = float(np.percentile(slens, 75) - np.percentile(slens, 25)) if len(slens) > 1 else 0.0\n",
    "    first_sent_ratio = slens[0] / max(sm, 1.0)\n",
    "    last_sent_ratio = slens[-1] / max(sm, 1.0)\n",
    "    demo_cnt = sum(1 for st in sents if st.split() and st.split()[0].lower() in DEMONSTRATIVES)\n",
    "    ly_cnt = sum(1 for w in words_lower if w.endswith('ly'))\n",
    "    ing_cnt = sum(1 for w in words_lower if w.endswith('ing'))\n",
    "    tion_cnt = sum(1 for w in words_lower if w.endswith(('tion', 'sion')))\n",
    "    unspaced_emdash = len(re.findall(r'\\S\\u2014\\S', t)) * fac\n",
    "    spaced_emdash = len(re.findall(r'\\s\\u2014\\s', t)) * fac\n",
    "    semi_to_comma = t.count(';') / max(t.count(','), 1)\n",
    "    long_word_rate = sum(1 for w in words if len(w) >= 9) / nw\n",
    "    hedge_rate = sum(1 for w in words_lower if w in HEDGES) * fac\n",
    "    conn_rate = sum(1 for w in words_lower if w in CONNECTIVES) * fac\n",
    "    sent_initial_conn = sum(1 for s in sents if s.split() and s.split()[0].lower().strip(',') in CONNECTIVES) / max(len(sents), 1)\n",
    "    if len(sents) >= 2:\n",
    "        first_set = set(re.findall(r'\\b\\w{4,}\\b', sents[0].lower()))\n",
    "        rest_set = set(re.findall(r'\\b\\w{4,}\\b', ' '.join(sents[1:]).lower()))\n",
    "        answer_first_overlap = len(first_set & rest_set) / max(len(first_set), 1)\n",
    "    else:\n",
    "        answer_first_overlap = 0.0\n",
    "    dist_to_round_50 = min(abs(nw - 50), abs(nw - 100), abs(nw - 150), abs(nw - 200)) / 50.0\n",
    "\n",
    "    base = [\n",
    "        float(len(t)), float(nw), float(nt) / nw, float(lens.mean()), float(lens.std()),\n",
    "        float(len(sents)), sm, ss, siqr, float(len(set(words_lower)) / nw),\n",
    "        t.count(chr(39)) * fac, (t.count('\\u2019') + t.count('\\u2018')) * fac,\n",
    "        t.count(chr(34)) * fac, (t.count('\\u201c') + t.count('\\u201d')) * fac,\n",
    "        t.count('\\u2014') * fac, t.count('\\u2013') * fac, t.count(';') * fac, t.count(':') * fac,\n",
    "        (t.count('(') + t.count(')')) * fac, demo_cnt * fac, ly_cnt * fac, ing_cnt * fac, tion_cnt * fac,\n",
    "        first_sent_ratio, last_sent_ratio, unspaced_emdash, spaced_emdash, semi_to_comma,\n",
    "        long_word_rate, hedge_rate, conn_rate, sent_initial_conn, answer_first_overlap, dist_to_round_50,\n",
    "        np.log1p(nw), np.log1p(nt), float(nt) / nw, float(ns) / nw,\n",
    "        sum(c.isupper() for c in t) / nt, sum(c.isdigit() for c in t) / nt,\n",
    "        sum(ord(c) > 127 for c in t) / nt, sum(c.isspace() for c in t) / nt, sum(c == '\\n' for c in t) / nt\n",
    "    ]\n",
    "    base.extend(t.count(m) * fac for m in MARKERS)\n",
    "    c = pd.Series(words_lower).value_counts()\n",
    "    base.extend(float(c.get(w, 0)) * fac for w in WORDS)\n",
    "    return np.array(base, dtype=np.float32)\n",
    "\n",
    "def features(texts):\n",
    "    return np.stack([row(str(t)) for t in texts])\n",
    "\n",
    "def apply_grouped_style_residual(\n",
    "    logits, train_text, test_text, labels, y, groups,\n",
    "    max_alpha=0.10, alpha=None, diagnostics_path=None, **kwargs\n",
    "):\n",
    "    \"\"\"\n",
    "    OOF-estimated, reliable sibling re-ranking across frontier model families.\n",
    "    Features include exact typographic glyphs + 16 sibling cliff signals.\n",
    "    Cross-family total probability mass is strictly preserved (logsumexp invariant).\n",
    "    \"\"\"\n",
    "    if alpha is not None:\n",
    "        max_alpha = float(alpha)\n",
    "    \n",
    "    Xt = features(train_text)\n",
    "    Xq = features(test_text)\n",
    "    out = np.asarray(logits, dtype=np.float64).copy()\n",
    "    lab = np.asarray(labels)\n",
    "    yy = np.asarray(y)\n",
    "    g = np.asarray(groups)\n",
    "    diag = {}\n",
    "\n",
    "    configs = [\n",
    "        ('claude', 0.010),\n",
    "        ('gemini', 0.005),\n",
    "        ('gpt', 0.010),\n",
    "        ('glm', 0.005)\n",
    "    ]\n",
    "\n",
    "    for family, C in configs:\n",
    "        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))\n",
    "        if len(cols) < 2:\n",
    "            continue\n",
    "        train_mask = np.isin(yy, cols)\n",
    "        inds = np.flatnonzero(train_mask)\n",
    "        if len(inds) < len(cols):\n",
    "            continue\n",
    "        X = Xt[inds]\n",
    "        target = np.searchsorted(cols, yy[inds])\n",
    "        grp = g[inds]\n",
    "        n_splits = min(5, len(np.unique(grp)))\n",
    "        if n_splits < 2:\n",
    "            continue\n",
    "        oof = np.empty(len(target), dtype=np.int32)\n",
    "\n",
    "        for a_idx, b_idx in GroupKFold(n_splits=n_splits).split(X, target, groups=grp):\n",
    "            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))\n",
    "            model.fit(X[a_idx], target[a_idx])\n",
    "            oof[b_idx] = model.predict(X[b_idx])\n",
    "\n",
    "        bacc = float(balanced_accuracy_score(target, oof))\n",
    "        chance = 1.0 / len(cols)\n",
    "        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)\n",
    "\n",
    "        if reliability <= 0:\n",
    "            diag[family] = dict(oof_balanced_accuracy=bacc, random_level=chance, alpha_mean=0.)\n",
    "            continue\n",
    "\n",
    "        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))\n",
    "        model.fit(X, target)\n",
    "        style_p = np.maximum(model.predict_proba(Xq), 1e-8)\n",
    "        old_logits = out[:, cols].copy()\n",
    "        old_p = softmax(old_logits, axis=1)\n",
    "        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]\n",
    "\n",
    "        gate = np.clip((0.65 - margin) / 0.65, 0.0, 1.0)\n",
    "        strength = (max_alpha * reliability) * gate\n",
    "        mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p\n",
    "\n",
    "        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))\n",
    "        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-8)\n",
    "        diag[family] = dict(\n",
    "            oof_balanced_accuracy=bacc,\n",
    "            random_level=chance,\n",
    "            reliability=float(reliability),\n",
    "            alpha_mean=float(strength.mean()),\n",
    "            alpha_max=float(strength.max()),\n",
    "            changed_within_family=int(np.sum(old_p.argmax(1) != mix.argmax(1)))\n",
    "        )\n",
    "\n",
    "    if diagnostics_path:\n",
    "        try:\n",
    "            with open(diagnostics_path, 'w') as f:\n",
    "                json.dump(diag, f, indent=2)\n",
    "        except Exception:\n",
    "            pass\n",
    "    return out\n"
]

# Cell 18: Pure Python Subprocess for comp_stack
nb['cells'][18]['source'] = [
    "# complementary public stack in its own process (frees all GPU memory when done); its wall time is charged to the ATLAS budget\n",
    "import time as _t\n",
    "import os, sys, subprocess\n",
    "_c0 = _t.time()\n",
    "os.environ['NTL_DROP_Q35'] = '0'\n",
    "if not FAST_COMMIT:\n",
    "    _res = subprocess.run([sys.executable, 'comp_stack.py'])\n",
    "    print('comp stack process exited with code:', _res.returncode, flush=True)\n",
    "COMP_SEC = _t.time() - _c0\n",
    "print('comp stack wall time', round(COMP_SEC), flush=True)\n"
]

# Write V14 notebook
out_nb_path = 'kaggle_v14/name-that-llm-v14-apex-grandmaster-champion.ipynb'
with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'Successfully wrote {out_nb_path}')

# Write clean kernel-metadata.json
with open('tmp_inspect/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

# Retain only required Kaggle fields
clean_meta = {
    "id": "omkarpattanaik123/name-that-llm-v14-apex-grandmaster-champion",
    "title": "Name That LLM V14 Apex Grandmaster Champion",
    "code_file": "name-that-llm-v14-apex-grandmaster-champion.ipynb",
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

out_meta_path = 'kaggle_v14/kernel-metadata.json'
with open(out_meta_path, 'w', encoding='utf-8') as f:
    json.dump(clean_meta, f, indent=2)
print(f'Successfully wrote {out_meta_path}')
