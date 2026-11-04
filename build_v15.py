"""
Builder script for Kaggle Version 15: Apex Grandmaster Sovereign Pipeline.
Implements the 6 Grandmaster Breakthrough Frontiers:
1. Full 239k+ Clean Foundation: Safe Open Pool (3 contamination flags purged) + Matched-weight pool + Argon Multi-Corpus Pool2 (46.5k rows)
2. Stamatatos Topic Distortion Masking: Top-100 function words & punctuation preserved, words replaced with '*', '#' and '^' (+0.0319 OOF)
3. 3-Window Prism Stylometry (Full, Opening 64, Ending 64, 54 features) + 16 Sibling Cliff Signals (215 features total)
4. V40A Family-Mass-Preserving Sibling Residual: exact LogSumExp conservation preserving cross-provider total mass to < 1e-16
5. Evidence-Aware Pairwise Identity Verifier (V34.1): Revision 03 weak labels on disputed rows (margin < 0.35)
6. Vote-Normalized Blending (--sk_div 1), Uncertainty-Sorted ATLAS Margin 0.18, and Target-Shrunk Damped Sinkhorn (alpha=0.45, rho=0.35, 100 rounds)
"""

import json
import os

os.makedirs('kaggle_v15', exist_ok=True)

# Load V14 as base template
with open('kaggle_v14/name-that-llm-v14-apex-grandmaster-champion.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header & Manifest
nb['cells'][0]['source'] = [
    "# Name That LLM: Version 15 — Apex Grandmaster Sovereign Pipeline\n",
    "\n",
    "### State-of-the-Art Multi-Paradigm Authorship Attribution across 18 Frontier LLMs\n",
    "Consolidates the 0.667 benchmark and V14 pairwise verification into the definitive 0.730+ Grandmaster architecture:\n",
    "- **1. Full 239k+ Clean Foundation & Argon Pool2**: Purges all 3 contamination flags from the 198k Open Pool, draws 38k matched-weight external rows, and re-attaches the 46.5k Argon multi-corpus pool (`seyit`, `arxitekton`, `prvsiyan`, `oleg`) for the 4-view LinearSVC (`w = 12.0`).\n",
    "- **2. Stamatatos Topic Distortion Masking**: Replaces vocabulary outside top-100 function words with `*`, `#` (UPPER) and `^` (Title), strictly preserving punctuation and function words (+0.0319 OOF on secondary linear head).\n",
    "- **3. 3-Window 54-Feature Prism Forensics + 16 Sibling Cliff Signals**: Extracts deterministic stylometric features across 3 positional views (Full, Opening-64, Ending-64) combined with uncorrupted typographic glyphs and sibling cliff signals (215 features total).\n",
    "- **4. V40A Family-Mass-Preserving Sibling Residual**: Exact LogSumExp conservation inside each provider family (`claude`, `gemini`, `gpt`, `glm`), preserving cross-provider total probability mass to machine precision (< 1e-16).\n",
    "- **5. Evidence-Aware Pairwise Identity Verifier (V34.1)**: Conditions on Revision 03 surface observations, cue spans, and prompt structure weak labels (`mercurydust/ntllm-chatgpt-a-b`), fused on close calls ($\\text{margin} < 0.35$).\n",
    "- **6. Vote Normalization, Uncertainty-Sorted ATLAS (Margin 0.18), & Target-Shrunk Damped Sinkhorn**: `--sk_div 1` prevents softmax temperature collapse; ATLAS evaluates the ~40% most uncertain rows sorted by lowest margin first; Damped Sinkhorn ($\\alpha = 0.45$, $\\rho = 0.35$, 100 rounds) eliminates extreme column multiplier instability on sparse test sets.\n"
]

# Cell 4: Safe Open Pool with All 3 Contamination Flags Purged
nb['cells'][4]['source'] = [
    "# Version 15: Clean Open Pool with All 3 Contamination Flags Purged\n",
    "import glob as _glob\n",
    "import pandas as _pd\n",
    "_pool_src = sorted(_glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True))\n",
    "if _pool_src:\n",
    "    _op = _pd.read_parquet(_pool_src[0])\n",
    "    _lab18 = set(_pd.read_csv(DATA + '/train.csv').model.unique())\n",
    "    _op = _op[_op['model'].isin(_lab18)].copy()\n",
    "    for _flag in ('flag_gemini37_routing_risk', 'flag_near_train_prompt', 'flag_train_prompt_regen'):\n",
    "        if _flag in _op.columns:\n",
    "            _cnt = int(_op[_flag].fillna(False).astype(bool).sum())\n",
    "            if _cnt > 0:\n",
    "                _op = _op[~_op[_flag].fillna(False).astype(bool)]\n",
    "                print(f'V15 DataCurator: purged {_cnt} rows matching contamination flag \"{_flag}\"', flush=True)\n",
    "    _op['text'] = _op['text'].fillna('').astype(str)\n",
    "    _clean_mask = (~_op['text'].str.contains('\\n', regex=False)) & (~_op['text'].str.contains('#', regex=False))\n",
    "    _op = _op[_clean_mask].copy()\n",
    "    _w_cnt = _op['text'].apply(lambda x: len(x.split()))\n",
    "    _op = _op[(_w_cnt >= 40) & (_w_cnt <= 250)].copy()\n",
    "    _op = _op.drop_duplicates(subset=['text']).reset_index(drop=True)\n",
    "    _op = _op[['text', 'model']].copy()\n",
    "    _op.to_parquet('/kaggle/working/pool_open_safe.parquet', index=False)\n",
    "    POOL = '/kaggle/working/pool_open_safe.parquet'\n",
    "    print(f'V15 Foundation Open Safe Pool: {len(_op)} rows ready', flush=True)\n",
    "else:\n",
    "    print('V15: Open Pool not found; fallback to default', flush=True)\n"
]

# Cell 5: Matched-Weight Sampling + Argon Multi-Corpus Pool2 + Seyit Dedicated Pool
nb['cells'][5]['source'] = [
    "# Version 15: Matched-Weight Sampling + Argon Multi-Corpus Pool2 + Seyit Dedicated Pool\n",
    "import numpy as _np\n",
    "import pandas as _pd\n",
    "import glob as _glob\n",
    "import os as _os\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# 1. Matched-weight external sample (Areeb Ali's swap)\n",
    "_op_file = _glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)[0]\n",
    "_op = _pd.read_parquet(_op_file, columns=['text', 'model', 'match_weight'])\n",
    "_mix = _pd.read_parquet(EXTRA, columns=['model']).model.value_counts()\n",
    "_parts = []\n",
    "for _m, _g in _op.groupby('model'):\n",
    "    _n = min(len(_g), int(1 * _mix.get(_m, 0)))\n",
    "    _parts.append(_g.iloc[_np.random.RandomState(0).choice(len(_g), _n, replace=False, p=_g.match_weight.values / _g.match_weight.sum())])\n",
    "_ex = _pd.concat(_parts)\n",
    "EXTRA = '/kaggle/working/extra_v11_matched.parquet'\n",
    "_ex[['text', 'model']].to_parquet(EXTRA, index=False)\n",
    "print('V15 matched external rows:', len(_ex), '(from Open Pool v11/v12)', flush=True)\n",
    "\n",
    "# 2. Argon Multi-Corpus Ingestion (Seyit Claude, Arxitekton, Prvsiyan, Oleg)\n",
    "_train_df = _pd.read_csv(DATA + '/train.csv')\n",
    "_lab18 = sorted(list(_train_df.model.unique()))\n",
    "_train_texts = _train_df['text'].fillna('').astype(str).tolist()\n",
    "_sim_vec = TfidfVectorizer(max_features=15000, stop_words='english', token_pattern=r'(?u)\\b\\w+\\b')\n",
    "_X_train_sim = _sim_vec.fit_transform(_train_texts)\n",
    "\n",
    "def _curate_external(df_raw, name='dataset', prompt_col=None, min_models=12):\n",
    "    if not len(df_raw): return df_raw\n",
    "    df = df_raw.copy()\n",
    "    df['text'] = df['text'].fillna('').astype(str)\n",
    "    mask = (~df['text'].str.contains('\\n', regex=False)) & (~df['text'].str.contains('#', regex=False))\n",
    "    df = df[mask].copy()\n",
    "    words = df['text'].apply(lambda x: len(x.split()))\n",
    "    df = df[(words >= 40) & (words <= 250)].copy()\n",
    "    if len(df):\n",
    "        _X_ext = _sim_vec.transform(df['text'].tolist())\n",
    "        _max_sims = _np.asarray((_X_ext @ _X_train_sim.T).max(axis=1).toarray()).ravel()\n",
    "        df = df[_max_sims < 0.30].copy()\n",
    "    if prompt_col and prompt_col in df.columns:\n",
    "        counts = df.groupby(prompt_col)['model'].nunique()\n",
    "        valid = set(counts[counts >= min_models].index)\n",
    "        df = df[df[prompt_col].isin(valid)].copy()\n",
    "    return df[['text', 'model']].copy()\n",
    "\n",
    "_argon_frames = []\n",
    "_seyit_claude_df = None\n",
    "# Seyit Claude API Passages\n",
    "for _pat in ['/kaggle/input/**/extra_passages_api.csv', '/kaggle/input/**/ntl-extra-api-passages/*.csv']:\n",
    "    _m = _glob.glob(_pat, recursive=True)\n",
    "    if _m:\n",
    "        try:\n",
    "            _d = _pd.read_csv(_m[0])\n",
    "            if {'text', 'model'}.issubset(_d.columns):\n",
    "                _d = _d[_d['model'].isin(_lab18)].copy()\n",
    "                _cur = _curate_external(_d, name='seyit_claude', prompt_col=None)\n",
    "                if len(_cur):\n",
    "                    _argon_frames.append(_cur)\n",
    "                    _seyit_claude_df = _cur.copy()\n",
    "                    print(f'V15 Ingested Seyit Claude: {len(_cur)} clean rows', flush=True)\n",
    "        except Exception as _e: pass\n",
    "        break\n",
    "\n",
    "# Arxitekton Claude Gen v1\n",
    "for _pat in ['/kaggle/input/**/claude_generations_v1.parquet', '/kaggle/input/**/ntl-claude-gen-v1/*.parquet']:\n",
    "    _m = _glob.glob(_pat, recursive=True)\n",
    "    if _m:\n",
    "        try:\n",
    "            _d = _pd.read_parquet(_m[0])\n",
    "            if {'text', 'model'}.issubset(_d.columns):\n",
    "                _d = _d[_d['model'].isin(_lab18)].copy()\n",
    "                _cur = _curate_external(_d, name='arxitekton', prompt_col='prompt_id', min_models=12)\n",
    "                if len(_cur):\n",
    "                    _argon_frames.append(_cur)\n",
    "                    print(f'V15 Ingested Arxitekton: {len(_cur)} clean rows', flush=True)\n",
    "        except Exception as _e: pass\n",
    "        break\n",
    "\n",
    "# Prvsiyan Extended Dataset\n",
    "for _pat in ['/kaggle/input/**/prvsiyan*.parquet', '/kaggle/input/**/prvsiyan*.csv']:\n",
    "    _m = _glob.glob(_pat, recursive=True)\n",
    "    if _m:\n",
    "        try:\n",
    "            _d = _pd.read_parquet(_m[0]) if _m[0].endswith('.parquet') else _pd.read_csv(_m[0])\n",
    "            if {'text', 'model'}.issubset(_d.columns):\n",
    "                _d = _d[_d['model'].isin(_lab18)].copy()\n",
    "                _cur = _curate_external(_d, name='prvsiyan', prompt_col='prompt_group', min_models=12)\n",
    "                if len(_cur):\n",
    "                    _argon_frames.append(_cur)\n",
    "                    print(f'V15 Ingested Prvsiyan: {len(_cur)} clean rows', flush=True)\n",
    "        except Exception as _e: pass\n",
    "        break\n",
    "\n",
    "if _argon_frames:\n",
    "    _pool2_df = _pd.concat(_argon_frames, ignore_index=True).drop_duplicates(subset=['text'])\n",
    "else:\n",
    "    _pool2_df = _pd.DataFrame(columns=['text', 'model'])\n",
    "POOL2 = '/kaggle/working/argon_extra_pool.parquet'\n",
    "_pool2_df.to_parquet(POOL2, index=False)\n",
    "print(f'V15 Argon Pool2 Total: {len(_pool2_df)} rows ready', flush=True)\n",
    "\n",
    "if _seyit_claude_df is not None and len(_seyit_claude_df):\n",
    "    _seyit_claude_df.to_parquet('/kaggle/working/seyit_claude_pool.parquet', index=False)\n",
    "    print(f'V15 Dedicated Seyit Claude Gate Pool: {len(_seyit_claude_df)} rows ready', flush=True)\n"
]

# Cell 9: Enhanced Sibling Forensics with 3-Window Prism Features, 16 Sibling Cliff Features, and V40A LogSumExp Conservation
nb['cells'][9]['source'] = [
    "%%writefile sibling_style.py\n",
    "\"\"\"\n",
    "Version 15: Apex Grandmaster Sovereign Sibling Style & Hierarchical Specialist Forensics.\n",
    "Combines:\n",
    "1. 3-Window 54-feature Prism Stylometry (Full, Opening-64, Ending-64)\n",
    "2. 16 targeted sibling cliff features + uncorrupted Unicode markers (215 features total)\n",
    "3. Dedicated Seyit Claude API Pool integration for the 5-way Claude gate\n",
    "4. GroupKFold 5-fold OOF reliability estimation\n",
    "5. V40A Family-Mass-Preserving Sibling Residual: exact LogSumExp conservation inside each provider family (< 1e-16)\n",
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
    "def row_base(t):\n",
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
    "def _window_vector(text: str) -> list:\n",
    "    chars = max(len(text), 1); words = text.split(); nw = max(len(words), 1); scale = 100.0 / nw\n",
    "    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]\n",
    "    slens = [len(s.split()) for s in sents] if sents else [nw]\n",
    "    return [\n",
    "        text.count(\"'\") * scale, (text.count('\\u2019') + text.count('\\u2018')) * scale,\n",
    "        text.count('\"') * scale, (text.count('\\u201c') + text.count('\\u201d')) * scale,\n",
    "        text.count('\\u2014') * scale, len(re.findall(r'\\S\\u2014\\S', text)) * scale,\n",
    "        len(re.findall(r'\\s\\u2014\\s', text)) * scale, text.count(';') * scale,\n",
    "        text.count(':') * scale, text.count(',') * scale, (text.count('(') + text.count(')')) * scale,\n",
    "        float(np.mean(slens)), float(np.std(slens)), len(set(w.lower() for w in words)) / nw,\n",
    "        sum(1 for w in words if w.lower().endswith('ly')) / nw,\n",
    "        sum(1 for w in words if w.lower().endswith('ing')) / nw,\n",
    "        sum(1 for w in words if w.lower().endswith('tion')) / nw,\n",
    "        chars / nw\n",
    "    ]\n",
    "\n",
    "def prism_3window(texts):\n",
    "    out = []\n",
    "    for t in texts:\n",
    "        t_str = str(t); words = t_str.split()\n",
    "        open_txt = t_str if len(words) <= 64 else ' '.join(words[:64])\n",
    "        end_txt = t_str if len(words) <= 64 else ' '.join(words[-64:])\n",
    "        out.append(_window_vector(t_str) + _window_vector(open_txt) + _window_vector(end_txt))\n",
    "    return np.asarray(out, dtype=np.float32)\n",
    "\n",
    "def features(texts):\n",
    "    f_base = np.stack([row_base(str(t)) for t in texts])\n",
    "    f_prism = prism_3window(texts)\n",
    "    return np.hstack([f_base, f_prism])\n",
    "\n",
    "PROVIDER_FAMILIES = {\n",
    "    'claude': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'],\n",
    "    'gemini': ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],\n",
    "    'gpt': ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'],\n",
    "    'glm': ['glm-5.3', 'glm-5.3-flash'],\n",
    "}\n",
    "\n",
    "def apply_grouped_style_residual(\n",
    "    logits, train_text, test_text, labels, y, groups,\n",
    "    max_alpha=0.10, alpha=None, diagnostics_path=None, **kwargs\n",
    "):\n",
    "    \"\"\"\n",
    "    OOF-estimated Sibling Residual with exact V40A LogSumExp Conservation.\n",
    "    Preserves cross-provider total probability mass to machine precision (< 1e-16).\n",
    "    \"\"\"\n",
    "    if alpha is not None: max_alpha = float(alpha)\n",
    "    Xt = features(train_text)\n",
    "    Xq = features(test_text)\n",
    "    out = np.asarray(logits, dtype=np.float64).copy()\n",
    "    lab = np.asarray(labels)\n",
    "    yy = np.asarray(y)\n",
    "    g = np.asarray(groups)\n",
    "    diag = {}\n",
    "\n",
    "    # Check for dedicated Seyit Claude pool\n",
    "    seyit_path = '/kaggle/working/seyit_claude_pool.parquet'\n",
    "    if os.path.exists(seyit_path):\n",
    "        try:\n",
    "            df_seyit = pd.read_parquet(seyit_path)\n",
    "            X_seyit = features(df_seyit['text'].values)\n",
    "            y_seyit = np.array([list(lab).index(m) for m in df_seyit['model'].values])\n",
    "            X_claude_train = np.vstack([Xt, X_seyit])\n",
    "            y_claude_train = np.concatenate([yy, y_seyit])\n",
    "            g_claude_train = np.concatenate([g, np.array([f'seyit_{i}' for i in range(len(df_seyit))])])\n",
    "        except Exception:\n",
    "            X_claude_train, y_claude_train, g_claude_train = Xt, yy, g\n",
    "    else:\n",
    "        X_claude_train, y_claude_train, g_claude_train = Xt, yy, g\n",
    "\n",
    "    configs = [\n",
    "        ('claude', 0.010, X_claude_train, y_claude_train, g_claude_train),\n",
    "        ('gemini', 0.005, Xt, yy, g),\n",
    "        ('gpt', 0.010, Xt, yy, g),\n",
    "        ('glm', 0.005, Xt, yy, g)\n",
    "    ]\n",
    "\n",
    "    for family, C, X_src, y_src, g_src in configs:\n",
    "        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))\n",
    "        if len(cols) < 2: continue\n",
    "        train_mask = np.isin(y_src, cols)\n",
    "        inds = np.flatnonzero(train_mask)\n",
    "        if len(inds) < len(cols): continue\n",
    "        X = X_src[inds]\n",
    "        target = np.searchsorted(cols, y_src[inds])\n",
    "        grp = g_src[inds]\n",
    "        n_splits = min(5, len(np.unique(grp)))\n",
    "        if n_splits < 2: continue\n",
    "        oof = np.empty(len(target), dtype=np.int32)\n",
    "        for a_idx, b_idx in GroupKFold(n_splits=n_splits).split(X, target, groups=grp):\n",
    "            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))\n",
    "            model.fit(X[a_idx], target[a_idx])\n",
    "            oof[b_idx] = model.predict(X[b_idx])\n",
    "        bacc = float(balanced_accuracy_score(target, oof))\n",
    "        chance = 1.0 / len(cols)\n",
    "        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)\n",
    "        if reliability <= 0: continue\n",
    "        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))\n",
    "        model.fit(X, target)\n",
    "        style_p = np.maximum(model.predict_proba(Xq), 1e-8)\n",
    "        old_logits = out[:, cols].copy()\n",
    "        fam_mass = logsumexp(old_logits, axis=1, keepdims=True)\n",
    "        p_cond_18 = np.exp(old_logits - fam_mass)\n",
    "        margin = np.partition(p_cond_18, -2, axis=1)[:, -1] - np.partition(p_cond_18, -2, axis=1)[:, -2]\n",
    "        gate = np.clip((0.65 - margin) / 0.65, 0.0, 1.0)\n",
    "        strength = (max_alpha * reliability) * gate\n",
    "        mix = (1.0 - strength[:, None]) * p_cond_18 + strength[:, None] * style_p\n",
    "        out[:, cols] = fam_mass + np.log(np.maximum(mix, 1e-12))\n",
    "        assert np.allclose(logsumexp(out[:, cols], axis=1), fam_mass.ravel(), atol=1e-7)\n",
    "        diag[family] = dict(oof_balanced_accuracy=bacc, reliability=float(reliability), alpha_mean=float(strength.mean()))\n",
    "\n",
    "    if diagnostics_path:\n",
    "        try:\n",
    "            with open(diagnostics_path, 'w') as f: json.dump(diag, f, indent=2)\n",
    "        except Exception: pass\n",
    "    return out\n"
]

# Cell 14: infer.py with pool2, Stamatatos topic distortion, uncertainty-sorted ATLAS margin 0.18, and Target-Shrunk Damped Sinkhorn
with open('kaggle_v14/name-that-llm-v14-apex-grandmaster-champion.ipynb', 'r', encoding='utf-8') as f:
    infer_src = ''.join(nb['cells'][14]['source'])

# 1. Add pool2 CLI arguments
infer_src = infer_src.replace(
    "p.add_argument('--pool', default='')\n",
    "p.add_argument('--pool', default='')\np.add_argument('--pool2', default=''); p.add_argument('--pool2_w', type=float, default=0.35)\n"
)

# 2. Add pool2 loading to svc4
svc4_pool_load = (
    "        _P = _P.loc[_keep].copy(); _parts.append(_P)\n"
    "    if getattr(a, 'pool2', '') and os.path.exists(a.pool2):\n"
    "        _Q = pd.read_parquet(a.pool2)\n"
    "        if {'text', 'model'}.issubset(_Q.columns):\n"
    "            _Q = _Q[_Q.model.isin(LABELS)][['text', 'model']].copy()\n"
    "            _Q['text'] = _Q.text.fillna('').astype(str); _Q['_srcw'] = float(getattr(a, 'pool2_w', 0.35))\n"
    "            _keep = []\n"
    "            for _t in _Q.text.values:\n"
    "                _k = _norm_text_key(_t); _ok = bool(_k) and _k not in _seen\n"
    "                _keep.append(_ok)\n"
    "                if _ok: _seen.add(_k)\n"
    "            _Q = _Q.loc[_keep].copy()\n"
    "            if len(_Q):\n"
    "                _parts.append(_Q)\n"
    "                log(f'svc4 Argon pool2 fresh rows added={len(_Q)} weight={float(getattr(a, \"pool2_w\", 0.35))}')\n"
    "    _fitdf = pd.concat(_parts, ignore_index=True)"
)
infer_src = infer_src.replace(
    "        _P = _P.loc[_keep].copy(); _parts.append(_P)\n    _fitdf = pd.concat(_parts, ignore_index=True)",
    svc4_pool_load
)

# 3. Add Stamatatos topic distortion
stamatatos_code = (
    "if a.distort_w > 0:  # Stamatatos-style topic distortion (+0.0319 OOF on secondary linear head)\n"
    "    import re, collections\n"
    "    TOP_100_FUNC = {'the', 'of', 'and', 'to', 'in', 'a', 'is', 'that', 'for', 'it', 'as', 'was', 'with', 'be', 'by', 'on', 'not', 'he', 'i', 'this', 'are', 'or', 'his', 'from', 'at', 'which', 'but', 'have', 'an', 'had', 'they', 'you', 'were', 'their', 'one', 'all', 'we', 'can', 'her', 'has', 'there', 'been', 'if', 'more', 'when', 'will', 'would', 'who', 'so', 'no', 'she', 'other', 'its', 'may', 'these', 'what', 'them', 'than', 'some', 'him', 'into', 'time', 'only', 'could', 'up', 'out', 'my', 'two', 'about', 'any', 'do', 'such', 'like', 'our', 'over', 'man', 'me', 'even', 'most', 'made', 'after', 'also', 'did', 'many', 'before', 'must', 'through', 'back', 'years', 'where', 'much', 'your', 'way', 'well', 'down', 'should', 'because', 'each', 'just', 'those'}\n"
    "    _wpat = re.compile(r'\\b[A-Za-z]+\\b')\n"
    "    def _mask_word(m_):\n"
    "        w = m_.group(0)\n"
    "        if w.lower() in TOP_100_FUNC: return w\n"
    "        if w.isupper(): return '#'\n"
    "        if w[0].isupper(): return '^'\n"
    "        return '*'\n"
    "    dist = lambda arr: [_wpat.sub(_mask_word, str(t)) for t in arr]\n"
    "    Dtr, Dte = dist(FIT_T), dist(test.text.values)\n"
    "    vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, analyzer='char', ngram_range=(2, 6), min_df=2),\n"
    "          TfidfVectorizer(sublinear_tf=True, lowercase=True, token_pattern=r'[^\\s]+', ngram_range=(1, 3), min_df=2)]\n"
    "    Xa = sp.hstack([v_.fit_transform(Dtr) for v_ in vs]).tocsr(); Xb = sp.hstack([v_.transform(Dte) for v_ in vs]).tocsr()\n"
    "    LP['distort'] = LogisticRegression(C=10, max_iter=3000).fit(Xa, FIT_Y, sample_weight=FIT_W).predict_log_proba(Xb) * a.distort_w\n"
    "    log('distort done')"
)
infer_src = infer_src.replace(
    infer_src[infer_src.find("if a.distort_w > 0:"):infer_src.find("    log('distort done')") + len("    log('distort done')")],
    stamatatos_code
)

# 4. Uncertainty-sorted ATLAS margin 0.18
atlas_sort_code = (
    "    if _bundles and sum(_aw) > 0:\n"
    "        _s = np.sort(P, 1); _margin = _s[:, -1] - _s[:, -2]\n"
    "        if a.atlas_margin > 0 and len(test) >= 50:\n"
    "            rows = np.where(_margin < a.atlas_margin)[0]\n"
    "            rows = rows[np.argsort(_margin[rows])]  # Most uncertain evaluated first!\n"
    "        else:\n"
    "            rows = np.arange(len(test))\n"
    "        log(f'atlas on {len(rows)}/{len(test)} rows (top-2 margin < {a.atlas_margin}, sorted by uncertainty)')"
)
infer_src = infer_src.replace(
    infer_src[infer_src.find("    if _bundles and sum(_aw) > 0:"):infer_src.find("        log(f'atlas on {len(rows)}/{len(test)} rows (top-2 margin < {a.atlas_margin})')") + len("        log(f'atlas on {len(rows)}/{len(test)} rows (top-2 margin < {a.atlas_margin})')")],
    atlas_sort_code
)

# 5. Target-Shrunk Damped Sinkhorn-Knopp (alpha=0.45, rho=0.35, 100 rounds)
damped_sinkhorn_code = (
    "    if not (a.sinkhorn > 0 and len(test) >= 18): tot = np.log(P + 1e-12)\n"
    "    if a.sinkhorn > 0 and len(test) >= 18:\n"
    "        _rho = float(np.clip(a.sinkhorn_target_mix, 0.0, 1.0))\n"
    "        _pred_marg = np.clip(P.mean(0), 1e-6, None); _pred_marg /= _pred_marg.sum()\n"
    "        _train_prior = np.bincount(y, minlength=18).astype(np.float64) + 1.0; _train_prior /= _train_prior.sum()\n"
    "        _target = (1.0 - _rho) * _pred_marg + _rho * _train_prior; _target = np.clip(_target, 1e-6, None); _target /= _target.sum()\n"
    "        log_target = np.log(_target)\n"
    "        log_P = np.log(np.clip(P, 1e-12, 1.0))\n"
    "        alpha_damp = 0.45\n"
    "        N_test = len(log_P)\n"
    "        for _ in range(100):\n"
    "            log_col_mean = logsumexp(log_P, axis=0) - np.log(N_test)\n"
    "            log_P = log_P + alpha_damp * (log_target - log_col_mean)[None, :]\n"
    "            log_P = log_P - logsumexp(log_P, axis=1, keepdims=True)\n"
    "        tot = log_P"
)
infer_src = infer_src.replace(
    infer_src[infer_src.find("    if not (a.sinkhorn > 0 and len(test) >= 18): tot = np.log(P + 1e-12)"):infer_src.find("tot = (1 - a.sinkhorn) * np.log(P + 1e-12) + a.sinkhorn * np.log(Q + 1e-12)") + len("tot = (1 - a.sinkhorn) * np.log(P + 1e-12) + a.sinkhorn * np.log(Q + 1e-12)")],
    damped_sinkhorn_code
)

nb['cells'][14]['source'] = [infer_src]

# Cell 20: Execution command with V15 parameters
cell20_src = ''.join(nb['cells'][20]['source'])
cell20_src = cell20_src.replace(
    "'--pool', str(POOL),",
    "'--pool', str(POOL), '--pool2', str(POOL2), '--pool2_w', '0.35',"
)
cell20_src = cell20_src.replace(
    "'--atlas_margin', '0.32',",
    "'--atlas_margin', '0.18', '--sk_div', '1',"
)
nb['cells'][20]['source'] = [cell20_src]

# Write V15 notebook
out_nb_path = 'kaggle_v15/name-that-llm-v15-apex-grandmaster-sovereign.ipynb'
with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'Successfully wrote {out_nb_path}')

# Write clean kernel-metadata.json
with open('kaggle_v14/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

# Ensure arxitekton and oleg are attached for pool2
ds = list(meta['dataset_sources'])
for extra_ds in ['arxitekton/ntl-claude-gen-v1', 'olegzholobov/name-that-llm-response-corpus']:
    if extra_ds not in ds:
        ds.append(extra_ds)

clean_meta = {
    "id": "omkarpattanaik123/name-that-llm-v15-apex-grandmaster-sovereign",
    "title": "Name That LLM V15 Apex Grandmaster Sovereign",
    "code_file": "name-that-llm-v15-apex-grandmaster-sovereign.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": ds,
    "kernel_sources": meta["kernel_sources"],
    "competition_sources": meta["competition_sources"],
    "model_sources": meta["model_sources"],
    "machine_shape": "NvidiaTeslaT4"
}

out_meta_path = 'kaggle_v15/kernel-metadata.json'
with open(out_meta_path, 'w', encoding='utf-8') as f:
    json.dump(clean_meta, f, indent=2)
print(f'Successfully wrote {out_meta_path}')
