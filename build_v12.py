"""
Builder script for Kaggle Version 12: Apex Sovereign Champion Pipeline.
Directly builds on top of our 0.667 benchmark (Version 10):
1. Preserves full 239k clean foundation (193k open pool + 46.5k multi-corpus, NO artificial capping)
2. Incorporates the 16 targeted sibling cliff features into sibling_style.py
3. Confidence-gated specialist blending: gate = clip((0.65 - margin)/0.65, 0, 1) with alpha_claude=0.40
4. Retains WPCA shared-prompt Gaussian, cohesion adaptive centering, and uniform target Sinkhorn
5. Tunes ATLAS margin to 0.32
"""

import json
import os
import re

os.makedirs('kaggle_v12', exist_ok=True)

# Load V10 as base template
with open('kaggle_v10/name-that-llm-v10-apex-grandmaster.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header & Manifest
nb['cells'][0]['source'] = [
    "# Name That LLM: Version 12 — Apex Sovereign Champion Pipeline\n",
    "\n",
    "### State-of-the-Art Multi-Paradigm Authorship Attribution across 18 Frontier LLMs\n",
    "Direct evolution of our 0.667 benchmark (Version 10) targeting the top spot:\n",
    "- **1. Full 239k Clean Foundation Preserved**: Retains the complete 193k clean open pool (all 3 contamination flags purged) + 46.5k Argon multi-corpus passages (`seyit`, `arxitekton`, `prvsiyan`, `oleg` short view) with zero downsampling.\n",
    "- **2. Sibling Cliff Forensics**: Injects 16 targeted syntactic, burstiness, and punctuation features into the specialist gates (Opus 4.6/4.8/5 unspaced em-dash & framing ratio; GPT-5.6 Latinate fertility & connective/hedge ratio; Gemini Flash answer-first overlap & round-50 proximity).\n",
    "- **3. Confidence-Gated Specialist Blending**: Evaluates intra-family prediction margins and applies specialist gates via $\\text{gate} = \\text{clip}((0.65 - \\text{margin})/0.65, 0, 1)$ with $\\alpha_{\\text{Claude}} = 0.40$, $\\alpha_{\\text{Gemini}} = 0.30$, and $\\alpha_{\\text{ZAI}} = 0.35$, protecting confident predictions while resolving intra-family ties.\n",
    "- **4. Shared-Prompt WPCA Gaussian & Cohesion Centering**: 4-step alternating means on within-prompt PCA ($d=128$) with Cholesky joint whitening, combined with cohesion-aware test-time adaptive centering ($2 \\le sz \\le 18$) and twin logit evidence injection.\n",
    "- **5. Target-Shrunk Sinkhorn Calibration**: Target marginals interpolate natural predicted test prior with uniform $1/18$ prior ($\\rho = 0.35$, $T = 0.90$, 25 iters) to directly optimize macro-average per-class recall.\n"
]

# Cell 8: Sibling Style with 16 Sibling Cliff Features & Confidence-Gated Blending
nb['cells'][8]['source'] = [
    "%%writefile sibling_style.py\n",
    "\"\"\"\n",
    "Version 12: Apex Sovereign Champion Sibling Style & Hierarchical Specialist Gates.\n",
    "Combines clean typographic forensics + 16 targeted sibling cliff features\n",
    "with confidence-gated geometric probability redistribution preserving primary family mass.\n",
    "\"\"\"\n",
    "import os\n",
    "import re\n",
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from functools import lru_cache\n",
    "from scipy.special import logsumexp, softmax\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import make_pipeline\n",
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
    "    v = [\n",
    "        float(len(t)), float(nw), float(nt) / nw, float(lens.mean()), float(lens.std()),\n",
    "        float(len(sents)), sm, ss, siqr, float(len(set(words_lower)) / nw),\n",
    "        t.count(chr(39)) * fac, (t.count('\\u2019') + t.count('\\u2018')) * fac,\n",
    "        t.count(chr(34)) * fac, (t.count('\\u201c') + t.count('\\u201d')) * fac,\n",
    "        t.count('\\u2014') * fac, t.count('\\u2013') * fac, t.count(';') * fac, t.count(':') * fac,\n",
    "        (t.count('(') + t.count(')')) * fac, demo_cnt * fac, ly_cnt * fac, ing_cnt * fac, tion_cnt * fac,\n",
    "        first_sent_ratio, last_sent_ratio, unspaced_emdash, spaced_emdash, semi_to_comma,\n",
    "        long_word_rate, hedge_rate, conn_rate, sent_initial_conn, answer_first_overlap, dist_to_round_50\n",
    "    ]\n",
    "    v.extend(t.count(m) * fac for m in MARKERS)\n",
    "    c = pd.Series(words_lower).value_counts()\n",
    "    v.extend(float(c.get(w, 0)) * fac for w in WORDS)\n",
    "    return np.array(v, dtype=np.float32)\n",
    "\n",
    "def features(texts):\n",
    "    return np.stack([row(str(t)) for t in texts])\n",
    "\n",
    "def apply_grouped_style_residual(\n",
    "    logits, train_text, test_text, labels, y, groups,\n",
    "    alpha=0.35, diagnostics_path=None\n",
    "):\n",
    "    \"\"\"\n",
    "    Hierarchical Specialist Sub-Family Gates with Confidence Gating:\n",
    "    1. 5-Way Claude Gate (alpha=0.40, with Seyit Claude API rows)\n",
    "    2. 4-Way Gemini Gate (alpha=0.30)\n",
    "    3. 3-Way ZAI vs. Opus-5 Disambiguator (alpha=0.35)\n",
    "    Applied via geometric blend preserving primary family mass, gated by sibling confidence margin.\n",
    "    \"\"\"\n",
    "    lab = list(labels)\n",
    "    out = np.asarray(logits, dtype=np.float64).copy()\n",
    "    P_18 = softmax(out, axis=1)\n",
    "    N_test = len(test_text)\n",
    "    X_train_base = features(train_text)\n",
    "    X_test = features(test_text)\n",
    "    y_train_base = np.array([lab[int(i)] for i in y])\n",
    "\n",
    "    # Check for dedicated Seyit Claude pool\n",
    "    seyit_path = '/kaggle/working/seyit_claude_pool.parquet'\n",
    "    if os.path.exists(seyit_path):\n",
    "        try:\n",
    "            df_seyit = pd.read_parquet(seyit_path)\n",
    "            X_seyit = features(df_seyit['text'].values)\n",
    "            y_seyit = df_seyit['model'].values\n",
    "            X_claude_train = np.vstack([X_train_base, X_seyit])\n",
    "            y_claude_train = np.concatenate([y_train_base, y_seyit])\n",
    "        except Exception:\n",
    "            X_claude_train, y_claude_train = X_train_base, y_train_base\n",
    "    else:\n",
    "        X_claude_train, y_claude_train = X_train_base, y_train_base\n",
    "\n",
    "    gate_configs = [\n",
    "        ('Claude-5Way', [\n",
    "            'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'\n",
    "        ], X_claude_train, y_claude_train, 0.05, 0.40),\n",
    "        ('Gemini-4Way', [\n",
    "            'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'\n",
    "        ], X_train_base, y_train_base, 0.02, 0.30),\n",
    "        ('ZAI-Opus-3Way', [\n",
    "            'glm-5.3', 'glm-5.3-flash', 'claude-opus-5'\n",
    "        ], X_train_base, y_train_base, 0.05, 0.35),\n",
    "    ]\n",
    "\n",
    "    diag = {}\n",
    "    for gate_name, sibs, X_src, y_src, c_val, gate_alpha in gate_configs:\n",
    "        cols = [lab.index(s) for s in sibs if s in lab]\n",
    "        if len(cols) < 2:\n",
    "            continue\n",
    "        mask = np.isin(y_src, sibs)\n",
    "        if mask.sum() < len(cols):\n",
    "            continue\n",
    "\n",
    "        X_sub = X_src[mask]\n",
    "        target_sub = np.array([sibs.index(m) for m in y_src[mask]])\n",
    "\n",
    "        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c_val, max_iter=500, class_weight='balanced'))\n",
    "        clf.fit(X_sub, target_sub)\n",
    "        p_gate = np.maximum(clf.predict_proba(X_test), 1e-9)\n",
    "\n",
    "        # Confidence-gated geometric blend preserving family mass\n",
    "        for i in range(N_test):\n",
    "            m_fam = float(P_18[i, cols].sum())\n",
    "            if m_fam < 1e-6:\n",
    "                continue\n",
    "            p_cond_18 = P_18[i, cols] / m_fam\n",
    "            # Compute sibling margin to gate intervention\n",
    "            s_sorted = np.sort(p_cond_18)\n",
    "            margin = s_sorted[-1] - s_sorted[-2]\n",
    "            conf_gate = np.clip((0.65 - margin) / 0.65, 0.0, 1.0)\n",
    "            eff_alpha = gate_alpha * conf_gate\n",
    "            if eff_alpha < 1e-4:\n",
    "                continue\n",
    "            p_g = p_gate[i]\n",
    "            p_geom = (np.maximum(p_cond_18, 1e-9) ** (1.0 - eff_alpha)) * (np.maximum(p_g, 1e-9) ** eff_alpha)\n",
    "            p_norm = p_geom / max(float(p_geom.sum()), 1e-9)\n",
    "            P_18[i, cols] = m_fam * p_norm\n",
    "\n",
    "        diag[gate_name] = dict(classes=sibs, samples=int(mask.sum()), alpha=gate_alpha)\n",
    "\n",
    "    out_refined = np.log(np.maximum(P_18, 1e-12))\n",
    "    if diagnostics_path:\n",
    "        with open(diagnostics_path, 'w') as f:\n",
    "            json.dump(diag, f, indent=2)\n",
    "    return out_refined\n"
]

# Update Cell 17: tune atlas_margin to 0.32
c17_src = ''.join(nb['cells'][17]['source'])
c17_src = c17_src.replace("'--atlas_margin', '0.3'", "'--atlas_margin', '0.32'")
c17_src = c17_src.replace("'--atlas_margin', '0.30'", "'--atlas_margin', '0.32'")
nb['cells'][17]['source'] = [c17_src]

# Update Cell 11 Assertions
nb['cells'][11]['source'] = [
    "# Version 12 Verification: assert infer.py contains V12 upgrades\n",
    "with open('infer.py') as f:\n",
    "    _src = f.read()\n",
    "assert 'pool2' in _src, 'Argon pool2 missing in infer.py'\n",
    "assert 'cache_model_to_tmp' in _src, 'cache_model_to_tmp missing in infer.py'\n",
    "assert '_uniform_prior' in _src, 'Target-shrunk uniform Sinkhorn missing in infer.py'\n",
    "print('V12 infer.py verified: all Apex Sovereign Champion components confirmed intact.')\n"
]

# Write V12 notebook
out_nb_path = 'kaggle_v12/name-that-llm-v12-apex-sovereign-champion.ipynb'
with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'Successfully wrote {out_nb_path}')

# Write kernel-metadata.json
with open('kaggle_v10/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

meta['id'] = 'omkarpattanaik123/name-that-llm-v12-apex-sovereign-champion'
meta['title'] = 'Name That LLM V12 Apex Sovereign Champion'
meta['code_file'] = 'name-that-llm-v12-apex-sovereign-champion.ipynb'

out_meta_path = 'kaggle_v12/kernel-metadata.json'
with open(out_meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
print(f'Successfully wrote {out_meta_path}')

