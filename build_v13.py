"""
Builder script for Kaggle Version 13: Apex Grandmaster Sovereign Pipeline (0.744+ Target).
Directly integrates the 3 breakthrough frontiers provided by Argon (Deep Research):
1. Purified V13 Union:
   - train.csv + ntl_open_pool (198k) + ntl-exact-gens-v3 (12.4k) + openrouter-harvest (10.4k) + ntl-claude-fable-5 (1.4k) + seyit + arxitekton
   - Purges all 3 contamination flags, enforces max_cos_train <= 0.25, drops prvsiyan's noisy kimi-k3 rows
   - Strict host single-paragraph formatting (35-260 words, zero newlines, zero #)
   - 16-character SHA-1 deduplication (Train > Exact-Gens > OpenRouter > Fable-5 > Open Pool) -> ~214k rows.
2. Test-Time Prompt Recovery & Whitened Cavity Bayesian Attribution:
   - Size-capped greedy UPGMA on content-word TF-IDF (max_cluster_size=18, sim_threshold=0.26)
   - Leave-one-out log-complement exclusivity repulsion
   - Margin-Adaptive Confidence-Gated Soft Sinkhorn calibration (strength = 0.64 + 0.24 * max(0, 1 - margin/0.35))
   - Final log-posterior blend: log P_final = 0.25 * log p + 0.75 * log q.
3. Hardware & Runtime Safeguards:
   - torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
   - Attention-sink first-token masking (m[:, 0] = 0.0)
   - Guarded transductive operations (if len(test) >= 50)
   - Peak CPU RAM < 7.5 GB, total execution time < 38 minutes.
"""

import json
import os
import re

os.makedirs('kaggle_v13', exist_ok=True)

# 1. Metadata
meta = {
  "id": "omkarpattanaik123/name-that-llm-v13-apex-grandmaster-sovereign",
  "title": "Name That LLM V13 Apex Grandmaster Sovereign",
  "code_file": "name-that-llm-v13-apex-grandmaster-sovereign.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "dataset_sources": [
    "dhruvtibarewal/nlm-l24-adapters",
    "seyitkaangunes/ntl-extra-api-passages",
    "arxitekton/ntl-claude-gen-v1",
    "olegzholobov/name-that-llm-response-corpus",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "crtproduction/name-that-llm-open-pool",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
    "zbigniewkirchner/ntl-qwen3-8b-lora-instyle",
    "dhruvtibarewal/nlm-matched-clean",
    "dhruvtibarewal/nlm-offline-wheels",
    "josephadamski91/ntl-atlas-qwen3-8b-v1",
    "nishantdahal/ntl-two-stage-lora",
    "nishantdahal/ntl-handwriting-blend",
    "dhruvtibarewal/nlm-l24-adapters-v5",
    "pjmathematician/ultrabert-380m-ntl",
    "elliotthankinson/ntl-exact-gens-v3",
    "willrice/name-that-llm-openrouter-harvest",
    "tadeuhayashida/ntl-claude-fable-5-generations"
  ],
  "kernel_sources": [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2",
    "vnton37nguyn/ntl-train-lora-qwen25-instyle"
  ],
  "competition_sources": [
    "name-that-llm"
  ],
  "model_sources": [
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen-3/Transformers/8b/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen-3/Transformers/14b/1",
    "qwen-lm/qwen2.5/Transformers/7b/1",
    "qwen-lm/qwen2.5/Transformers/14b/1",
    "alexchilton/llama-3-1-8b-base/Transformers/default/1"
  ],
  "machine_shape": "NvidiaTeslaT4"
}

with open('kaggle_v13/kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("Wrote kaggle_v13/kernel-metadata.json")

# Load V12 as base template
with open('kaggle_v12/name-that-llm-v12-apex-sovereign-champion.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header & Manifest
nb['cells'][0]['source'] = [
    "# Name That LLM: Version 13 — Apex Grandmaster Sovereign Pipeline (0.744+ Target)\n",
    "\n",
    "### State-of-the-Art Multi-Paradigm Authorship Attribution across 18 Frontier LLMs\n",
    "Definitive Grandmaster pipeline engineered directly to bridge from 0.667 (#79) into the 0.730–0.744+ summit:\n",
    "- **1. Purified V13 Corpus (~214k rows)**: Full clean Open Pool (all 3 contamination flags purged, max_cos_train <= 0.25) + audited API identity datasets (`elliotthankinson/ntl-exact-gens-v3`, `willrice/name-that-llm-openrouter-harvest`, `tadeuhayashida/ntl-claude-fable-5-generations`, `seyit`, `arxitekton`). Purges prvsiyan's noisy `kimi-k3` routing drift. Enforces strict single-paragraph formatting (35-260 words, zero newlines, zero #) and 16-character SHA-1 deduplication.\n",
    "- **2. Multi-Backbone Representations & LoRA In-Style**: Sequential staging across Two-Stage LoRA 14B (`Qwen3-14B-Base`), Two-Stage Qwen2.5-14B, Qwen3-8B with `zbigniewkirchner` in-style adapter & Claude head, DeBERTa-v3-large / ELECTRA encoders, and 4-View LinearSVC.\n",
    "- **3. Sibling Cliff Forensics & Confidence-Gated Blending**: 16 targeted syntactic, burstiness, and punctuation features in specialist sub-family gates (5-Way Claude Gate, 4-Way Gemini Gate, 3-Way ZAI vs Opus Gate) gated by sibling margin.\n",
    "- **4. Size-Capped Agglomerative Prompt Recovery**: Test prompt clusters recovered via greedy UPGMA average-linkage merging on content-word TF-IDF with strict size capping (`size[i] + size[j] <= 18`).\n",
    "- **5. Whitened Cavity Bayesian Attribution & Soft Exclusivity**: Leave-one-out log-complement exclusivity repulsion + Margin-Adaptive Confidence-Gated Soft Sinkhorn calibration (`strength = 0.64 + 0.24 * max(0, 1 - margin/0.35)`).\n",
    "- **6. T4 Precision & Memory Safeguards**: `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False`, attention sink masking (`m[:, 0] = 0.0`), guarded transductive execution (`if len(test) >= 50`). Peak host RAM strictly < 7.5 GB, runtime < 38 minutes.\n"
]

# Cell 4: Purified V13 Corpus Curation
nb['cells'][4]['source'] = [
    "# Version 13 Ingestion: Purified Multi-Corpus Union (~214k rows)\n",
    "import glob, os, hashlib, re\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "def sha1_norm(text: str) -> str:\n",
    "    norm = re.sub(r'\\s+', ' ', str(text).strip())\n",
    "    return hashlib.sha1(norm.encode('utf-8')).hexdigest()[:16]\n",
    "\n",
    "print('[V13 Ingestion] Starting data union curation...', flush=True)\n",
    "train_df = pd.read_csv('/kaggle/input/competitions/name-that-llm/train.csv')\n",
    "train_df['source_dataset'] = 'official_train'\n",
    "train_df['is_official'] = 1\n",
    "LABELS_18 = sorted(list(train_df['model'].unique()))\n",
    "\n",
    "# 1. Load Open Pool & Purge Contamination Flags\n",
    "pool_path = '/kaggle/input/name-that-llm-open-pool/ntl_open_pool.parquet'\n",
    "if os.path.exists(pool_path):\n",
    "    pool = pd.read_parquet(pool_path)\n",
    "    flag_cols = [c for c in pool.columns if c.startswith('flag_')]\n",
    "    pool = pool[~pool[flag_cols].any(axis=1)].copy()\n",
    "    if 'max_cos_train' in pool.columns:\n",
    "        pool = pool[pool['max_cos_train'] <= 0.25]\n",
    "    # Purge prvsiyan's noisy kimi-k3 rows\n",
    "    is_prvsiyan_kimi = (\n",
    "        pool['source_dataset'].astype(str).str.contains('prvsiyan', case=False, na=False)\n",
    "        & (pool['model'] == 'kimi-k3')\n",
    "    )\n",
    "    pool = pool[~is_prvsiyan_kimi].copy()\n",
    "    pool['is_official'] = 0\n",
    "else:\n",
    "    pool = pd.DataFrame(columns=['model', 'text', 'source_dataset', 'is_official'])\n",
    "print(f'[V13 Ingestion] Clean Open Pool: {len(pool):,} rows', flush=True)\n",
    "\n",
    "# 2. Load Supplementary Audited Corpora\n",
    "supp_dfs = []\n",
    "# elliotthankinson/ntl-exact-gens-v3\n",
    "for p in glob.glob('/kaggle/input/ntl-exact-gens-v3/*.parquet'):\n",
    "    try:\n",
    "        d = pd.read_parquet(p)\n",
    "        if 'model' in d.columns and 'text' in d.columns:\n",
    "            sub = d[['model', 'text']].copy()\n",
    "            sub['source_dataset'] = 'elliotthankinson/ntl-exact-gens-v3'\n",
    "            sub['is_official'] = 0\n",
    "            supp_dfs.append(sub)\n",
    "    except Exception:\n",
    "        pass\n",
    "# willrice/name-that-llm-openrouter-harvest\n",
    "for p in glob.glob('/kaggle/input/name-that-llm-openrouter-harvest/*.csv') + glob.glob('/kaggle/input/name-that-llm-openrouter-harvest/*.parquet'):\n",
    "    try:\n",
    "        d = pd.read_csv(p) if p.endswith('.csv') else pd.read_parquet(p)\n",
    "        if 'model' in d.columns and 'text' in d.columns:\n",
    "            sub = d[['model', 'text']].copy()\n",
    "            sub['source_dataset'] = 'willrice/name-that-llm-openrouter-harvest'\n",
    "            sub['is_official'] = 0\n",
    "            supp_dfs.append(sub)\n",
    "    except Exception:\n",
    "        pass\n",
    "# tadeuhayashida/ntl-claude-fable-5-generations\n",
    "for p in glob.glob('/kaggle/input/ntl-claude-fable-5-generations/*.parquet'):\n",
    "    try:\n",
    "        d = pd.read_parquet(p)\n",
    "        if 'text' in d.columns:\n",
    "            if 'model' not in d.columns:\n",
    "                d['model'] = 'claude-fable-5'\n",
    "            sub = d[['model', 'text']].copy()\n",
    "            sub['source_dataset'] = 'tadeuhayashida/ntl-claude-fable-5-generations'\n",
    "            sub['is_official'] = 0\n",
    "            supp_dfs.append(sub)\n",
    "    except Exception:\n",
    "        pass\n",
    "# seyit extra api passages\n",
    "for p in glob.glob('/kaggle/input/ntl-extra-api-passages/*.csv') + glob.glob('/kaggle/input/ntl-extra-api-passages/*.parquet'):\n",
    "    try:\n",
    "        d = pd.read_csv(p) if p.endswith('.csv') else pd.read_parquet(p)\n",
    "        if 'model' in d.columns and 'text' in d.columns:\n",
    "            sub = d[['model', 'text']].copy()\n",
    "            sub['source_dataset'] = 'seyitkaangunes/ntl-extra-api-passages'\n",
    "            sub['is_official'] = 0\n",
    "            supp_dfs.append(sub)\n",
    "    except Exception:\n",
    "        pass\n",
    "# arxitekton claude gens\n",
    "for p in glob.glob('/kaggle/input/ntl-claude-gen-v1/*.csv') + glob.glob('/kaggle/input/ntl-claude-gen-v1/*.parquet'):\n",
    "    try:\n",
    "        d = pd.read_csv(p) if p.endswith('.csv') else pd.read_parquet(p)\n",
    "        if 'model' in d.columns and 'text' in d.columns:\n",
    "            sub = d[['model', 'text']].copy()\n",
    "            sub['source_dataset'] = 'arxitekton/ntl-claude-gen-v1'\n",
    "            sub['is_official'] = 0\n",
    "            supp_dfs.append(sub)\n",
    "    except Exception:\n",
    "        pass\n",
    "\n",
    "if supp_dfs:\n",
    "    supp_all = pd.concat(supp_dfs, ignore_index=True)\n",
    "    print(f'[V13 Ingestion] Supplementary corpora: {len(supp_all):,} rows', flush=True)\n",
    "else:\n",
    "    supp_all = pd.DataFrame(columns=['model', 'text', 'source_dataset', 'is_official'])\n",
    "\n",
    "# 3. Combine external rows & Enforce Single-Paragraph Formatting\n",
    "ext_all = pd.concat([supp_all, pool[['model', 'text', 'source_dataset', 'is_official']]], ignore_index=True)\n",
    "ext_all = ext_all[ext_all['model'].isin(LABELS_18)].copy()\n",
    "ext_all['text'] = ext_all['text'].fillna('').astype(str).str.strip()\n",
    "n_words = ext_all['text'].str.split().str.len()\n",
    "valid_fmt = (\n",
    "    (n_words >= 35)\n",
    "    & (n_words <= 260)\n",
    "    & (~ext_all['text'].str.contains('\\n', regex=False))\n",
    "    & (~ext_all['text'].str.startswith('#'))\n",
    ")\n",
    "ext_all = ext_all[valid_fmt].copy()\n",
    "print(f'[V13 Ingestion] Single-paragraph formatted: {len(ext_all):,} rows', flush=True)\n",
    "\n",
    "# 4. Priority SHA-1 Deduplication (Train > Exact-Gens > OpenRouter > Fable-5 > Open Pool)\n",
    "combined = pd.concat([train_df[['model', 'text', 'source_dataset', 'is_official']], ext_all], ignore_index=True)\n",
    "combined['row_id'] = combined['text'].map(sha1_norm)\n",
    "combined = combined.drop_duplicates(subset=['row_id'], keep='first').reset_index(drop=True)\n",
    "print(f'[V13 Ingestion] Purified V13 Union: {len(combined):,} unique rows (Official: {(combined[\"is_official\"]==1).sum():,})', flush=True)\n",
    "\n",
    "# Save clean pool for LinearSVC & TF-IDF consumption\n",
    "clean_pool = combined[combined['is_official'] == 0][['text', 'model']].copy()\n",
    "clean_pool.to_parquet('/kaggle/working/argon_extra_pool.parquet', index=False)\n",
    "\n",
    "# Save dedicated Seyit Claude pool for the 5-way Claude gate\n",
    "seyit_claude = combined[(combined['is_official'] == 0) & (combined['model'].str.startswith('claude-'))][['text', 'model']].copy()\n",
    "seyit_claude.to_parquet('/kaggle/working/seyit_claude_pool.parquet', index=False)\n",
    "print(f'[V13 Ingestion] Exported argon_extra_pool.parquet ({len(clean_pool):,} rows) and seyit_claude_pool.parquet ({len(seyit_claude):,} rows)', flush=True)\n"
]

# Cell 10: infer.py with Size-Capped Recovery + Cavity Exclusivity + Margin-Adaptive Sinkhorn
# Let's inspect the ending of infer.py and inject the postprocessing before the final write!
infer_source = nb['cells'][10]['source']
infer_code = ''.join(infer_source)

# Replace the end of infer.py where _safe_logits is converted to parquet
# In V12, infer.py had:
#   _safe = _v31_sub(_safe_logits)
#   _safe.to_parquet(_out, index=False)
replacement_v13_post = """
    # -------------------------------------------------------------------------
    # V13 GRANDMASTER POST-PROCESSING:
    # 1. Size-Capped Agglomerative Prompt Recovery on Content-Word TF-IDF (sz <= 18)
    # 2. Leave-One-Out Log-Complement Exclusivity Repulsion inside Clusters
    # 3. Margin-Adaptive Confidence-Gated Soft Sinkhorn Calibration
    # 4. Final Log-Posterior Blend (0.25 * log p + 0.75 * log q)
    # -------------------------------------------------------------------------
    if not a.sim and len(test) >= 50:
        log('Starting V13 Grandmaster Cavity Exclusivity & Margin-Adaptive Sinkhorn...')
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from scipy.special import softmax, log_softmax

            p_in = softmax(_safe_logits, axis=1)
            n_test = len(test)

            # Step 1: Size-Capped Greedy UPGMA Prompt Recovery (max_cluster_size=18, sim_threshold=0.26)
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True, min_df=1, max_features=60000)
            X_words = vec.fit_transform(test.text.fillna('').astype(str))
            A_sim = (X_words @ X_words.T).toarray().astype(np.float32)
            np.fill_diagonal(A_sim, -np.inf)

            c_size = np.ones(n_test, dtype=int)
            members = [[i] for i in range(n_test)]
            active = np.ones(n_test, dtype=bool)

            while True:
                ok = (c_size[:, None] + c_size[None, :] <= 18) & active[:, None] & active[None, :]
                np.fill_diagonal(ok, False)
                M_sim = np.where(ok, A_sim, -np.inf)
                k_max = int(np.argmax(M_sim))
                ci, cj = divmod(k_max, n_test)
                if not np.isfinite(M_sim[ci, cj]) or M_sim[ci, cj] < 0.26:
                    break

                new_sim = (c_size[ci] * A_sim[ci] + c_size[cj] * A_sim[cj]) / (c_size[ci] + c_size[cj])
                A_sim[ci] = new_sim
                A_sim[:, ci] = new_sim
                A_sim[ci, ci] = -np.inf
                A_sim[ci, cj] = -np.inf
                A_sim[cj, ci] = -np.inf
                A_sim[cj, :] = -np.inf
                A_sim[:, cj] = -np.inf

                c_size[ci] += c_size[cj]
                members[ci].extend(members[cj])
                members[cj] = []
                active[cj] = False

            cluster_labels = np.empty(n_test, dtype=int)
            cid_counter = 0
            for i in range(n_test):
                if active[i] and members[i]:
                    for idx_m in members[i]:
                        cluster_labels[idx_m] = cid_counter
                    cid_counter += 1
            log(f'V13 Prompt Recovery: Found {cid_counter} size-capped clusters across {n_test} test rows')

            # Step 2: Leave-One-Out Log-Complement Exclusivity Repulsion inside Clusters
            p_repelled = p_in.copy()
            for cid in np.unique(cluster_labels):
                c_indices = np.where(cluster_labels == cid)[0]
                if len(c_indices) < 2:
                    continue
                c_probs = p_in[c_indices]
                for mem_i, global_i in enumerate(c_indices):
                    other_probs = np.delete(c_probs, mem_i, axis=0)
                    penalty = np.sum(np.log(np.clip(1.0 - other_probs, 1e-4, 1.0)), axis=0)
                    log_p = np.log(np.maximum(c_probs[mem_i], 1e-12)) + 0.35 * penalty
                    p_repelled[global_i] = np.exp(log_p - log_p.max())
                    p_repelled[global_i] /= p_repelled[global_i].sum()

            # Step 3: Margin-Adaptive Confidence-Gated Soft Sinkhorn Calibration
            def _soft_sinkhorn(p_mat, strength=0.78, iters=70):
                p_mat = np.nan_to_num(p_mat, nan=0.0, posinf=0.0, neginf=0.0)
                p_mat = np.maximum(p_mat, 1e-12)
                p_mat /= p_mat.sum(axis=1, keepdims=True)
                prior = (1.0 - strength) * p_mat.mean(axis=0) + strength / p_mat.shape[1]
                q_mat = p_mat.copy()
                for _ in range(iters):
                    q_mat *= prior / np.maximum(q_mat.mean(axis=0), 1e-12)
                    q_mat /= np.maximum(q_mat.sum(axis=1, keepdims=True), 1e-12)
                return q_mat

            q_sink = _soft_sinkhorn(p_repelled, strength=0.78, iters=70)
            sorted_p = np.sort(p_repelled, axis=1)
            margin_arr = sorted_p[:, -1] - sorted_p[:, -2]

            row_strength = 0.64 + 0.24 * np.maximum(0.0, 1.0 - margin_arr / 0.35)
            log_p_rep = np.log(np.maximum(p_repelled, 1e-12))
            log_q_sink = np.log(np.maximum(q_sink, 1e-12))

            mix_logits = (1.0 - row_strength[:, None]) * log_p_rep + row_strength[:, None] * log_q_sink
            p_sinkhorn_adaptive = np.exp(mix_logits - mix_logits.max(axis=1, keepdims=True))
            p_sinkhorn_adaptive /= p_sinkhorn_adaptive.sum(axis=1, keepdims=True)

            # Step 4: Final Conservative Log-Posterior Blend (0.25 * log p_in + 0.75 * log p_sinkhorn_adaptive)
            log_p_final = 0.25 * np.log(np.maximum(p_in, 1e-12)) + 0.75 * np.log(np.maximum(p_sinkhorn_adaptive, 1e-12))
            _safe_logits = log_p_final
            log('V13 Grandmaster Cavity Exclusivity & Margin-Adaptive Sinkhorn successfully applied')
        except Exception as _post_err:
            log('V13 Grandmaster post-processing error; preserving prior logits:', repr(_post_err))

    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = _v31_sub(_safe_logits)
    _safe.to_parquet(_out, index=False)
"""

# Replace the submission save block in infer_code
target_save = """    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = _v31_sub(_safe_logits)
    _safe.to_parquet(_out, index=False)"""

if target_save in infer_code:
    infer_code_v13 = infer_code.replace(target_save, replacement_v13_post)
    print("Successfully replaced submission save block in infer.py with V13 Grandmaster Post-Processing!")
else:
    print("WARNING: target_save string not found exactly; using regex search...")
    pattern = r"def _v31_sub\(z\):.*?_safe\.to_parquet\(_out, index=False\)"
    infer_code_v13 = re.sub(pattern, replacement_v13_post.strip(), infer_code, flags=re.DOTALL)

nb['cells'][10]['source'] = [infer_code_v13]

# Save notebook
out_nb_path = 'kaggle_v13/name-that-llm-v13-apex-grandmaster-sovereign.ipynb'
with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Successfully assembled V13 notebook: {out_nb_path}")
