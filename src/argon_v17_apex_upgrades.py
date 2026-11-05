"""
Version 17: Apex Grandmaster Breakthrough Upgrades.
Implements the 4 Frontiers to Bridge into the 0.740-0.747+ Global Top Tier:
1. Target-Class Specialist Deepening (12 High-Significance Features: Fable-5 Subordinators, MATTR-25, Flash-3.8 Answer-First Centrality, Kimi-K3 Rhythm).
2. Upgraded 227-Feature Stylometric Forensic Engine (161 base + 54 3-window prism + 12 specialist features).
3. Dual-Threshold Prompt Clustering & Centering (0.85 TF-IDF + 0.15 Embedding Distance; th0.62 Core + th0.72 Halo).
4. Prior-Balanced, Cluster-Constrained Transductive Pseudo-Labeling (Safe Stage-2 Self-Training).
"""

import os
import re
import json
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

# ==============================================================================
# 1. Target-Class Specialist Feature Extractor (12 Features)
# ==============================================================================
FABLE5_SUBORDINATORS = {
    "albeit", "insofar", "whereby", "nonetheless", "nevertheless", "conversely",
    "notwithstanding", "whereas", "wherein", "thereby", "henceforth"
}
FLASH38_SIGNPOSTS = {
    "first", "second", "third", "finally", "specifically", "importantly",
    "notably", "consequently", "in contrast", "for instance", "in summary"
}

def extract_low_recall_specialist_features(texts) -> np.ndarray:
    """
    Extracts 12 high-signal syntactic, lexical-richness, and structural-organization
    features engineered specifically for the 4 lowest-recall competition classes:
      - claude-fable-5 (high-effort adaptive thinking subordinators & nested clauses)
      - gpt-5.6-luna (nano-tier MATTR-25 lexical compression & verb repetition)
      - gemini-3.8-flash (answer-first sentence centrality & signpost enumeration)
      - kimi-k3 (bilingual-pretrained clause symmetry & comma-to-period rhythm)
    """
    out = []
    for t in texts:
        s = str(t).strip()
        words = re.findall(r"\b[A-Za-z]+\b", s)
        w_lower = [w.lower() for w in words]
        nw = max(len(w_lower), 1)
        scale = 100.0 / nw

        sents = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", s) if seg.strip()]
        ns = max(len(sents), 1)

        # --- A. claude-fable-5 features ---
        fable_sub_rate = sum(1 for w in w_lower if w in FABLE5_SUBORDINATORS) * scale
        nominal_suffix_rate = sum(
            1 for w in w_lower if len(w) >= 7 and w.endswith(("tion", "ment", "ity", "ance", "ence"))
        ) / nw
        nested_clause_rate = (s.count("(") + s.count("\u2014") + s.count(";")) / ns

        # --- B. gpt-5.6-luna features (MATTR-25 & Hapax Legomena ratio) ---
        win = 25
        if nw >= win:
            mattr25 = float(np.mean([len(set(w_lower[i:i + win])) / float(win) for i in range(nw - win + 1)]))
        else:
            mattr25 = len(set(w_lower)) / float(nw)
        counts = pd.Series(w_lower).value_counts()
        hapax_ratio = float((counts == 1).sum()) / float(nw)
        rare_long_word_ratio = sum(1 for w in w_lower if len(w) >= 10) / float(nw)

        # --- C. gemini-3.8-flash features (Answer-First Centrality & Signposts) ---
        if len(sents) >= 2:
            s0_words = set(w for w in re.findall(r"\b[a-z]{4,}\b", sents[0].lower()))
            body_words = set(w for w in re.findall(r"\b[a-z]{4,}\b", " ".join(sents[1:]).lower()))
            answer_first_centrality = len(s0_words & body_words) / float(max(len(s0_words), 1))
        else:
            answer_first_centrality = 0.0

        signpost_hits = sum(
            1 for seg in sents for sp in FLASH38_SIGNPOSTS
            if seg.lower().startswith(sp + ",") or (f", {sp}," in seg.lower())
        ) / float(ns)
        round10_dev = abs(nw - round(nw / 10.0) * 10.0) / 10.0

        # --- D. kimi-k3 features (Clause symmetry & comma rhythm) ---
        commas_per_sent = s.count(",") / float(ns)
        clause_parts = [len(c.split()) for seg in sents for c in seg.split(",") if c.strip()]
        clause_len_cv = float(np.std(clause_parts) / max(np.mean(clause_parts), 1.0)) if len(clause_parts) > 1 else 0.0
        straight_vs_curly_ind = 1.0 if s.count("'") > (s.count("\u2019") + s.count("\u2018")) else 0.0

        out.append([
            fable_sub_rate, nominal_suffix_rate, nested_clause_rate,
            mattr25, hapax_ratio, rare_long_word_ratio,
            answer_first_centrality, signpost_hits, round10_dev,
            commas_per_sent, clause_len_cv, straight_vs_curly_ind
        ])
    return np.asarray(out, dtype=np.float32)


# ==============================================================================
# 2. Dual-Threshold Prompt Clustering (th0.62 Core + th0.72 Halo, tfidf 0.85)
# ==============================================================================
def compute_dual_threshold_prompt_clusters(
    test_texts,
    E_te: np.ndarray,
    tfidf_weight: float = 0.85,
    th_core: float = 0.62,
    th_halo: float = 0.72
) -> tuple:
    """
    Computes Core and Halo prompt clusters using 0.85 TF-IDF + 0.15 normalized embedding distance.
    Returns:
      g_core: Cluster assignments for tight prompt core (th = 0.62, >94% precision)
      g_halo: Cluster assignments for broad prompt halo (th = 0.72, ~85% recall)
    """
    n_te = len(test_texts)
    if n_te < 18:
        return np.arange(n_te), np.arange(n_te)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
        max_features=60000
    )
    X_tf = vec.fit_transform([str(t) for t in test_texts])
    D_tf = np.clip(1.0 - (X_tf @ X_tf.T).toarray(), 0.0, 1.0)

    if E_te is not None and len(E_te) == n_te:
        Z_norm = E_te / (np.linalg.norm(E_te, axis=1, keepdims=True) + 1e-12)
        D_emb = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, 1.0)
        D = tfidf_weight * D_tf + (1.0 - tfidf_weight) * D_emb
    else:
        D = D_tf

    np.fill_diagonal(D, 0.0)
    Z_link = linkage(squareform(D, checks=False), method="average")
    g_core = fcluster(Z_link, t=th_core, criterion="distance")
    g_halo = fcluster(Z_link, t=th_halo, criterion="distance")

    return g_core, g_halo


# ==============================================================================
# 3. Safe Transductive Stage-2 Pseudo-Labeling (Prior-Balanced & Cluster-Capped)
# ==============================================================================
def select_safe_transductive_pseudo_labels(
    probs_te: np.ndarray,
    cluster_ids_te: np.ndarray,
    n_train_per_class: float = 150.0,
    pseudo_mass_share: float = 0.20,
    min_top2_margin: float = 0.45,
    max_norm_entropy: float = 0.22,
    max_per_class_quota: int = 40
) -> tuple:
    """
    Selects high-confidence test passages for Stage-2 transductive readout refitting
    while strictly preventing confirmation bias and cluster error propagation:
      1. Requires top-2 margin >= 0.45 and normalized entropy <= 0.22.
      2. Enforces at most ONE passage per (prompt_cluster, predicted_class) pair.
      3. Caps per-class quota and assigns class-balanced sample weights so each of
         the 18 models retains exact equal prior mass (no majority-class runaway).
    """
    N, K = probs_te.shape
    if N < 50:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=np.float32)

    P = np.clip(probs_te, 1e-12, 1.0)
    P /= P.sum(axis=1, keepdims=True)

    pred_cls = np.argmax(P, axis=1)
    sorted_p = np.sort(P, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    norm_ent = -np.sum(P * np.log(P), axis=1) / np.log(float(K))

    eligible = np.where((margin >= min_top2_margin) & (norm_ent <= max_norm_entropy))[0]
    if len(eligible) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=np.float32)

    eligible = eligible[np.argsort(-margin[eligible])]
    seen_cluster_class = set()
    dedup_idx = []
    for idx in eligible:
        key = (int(cluster_ids_te[idx]), int(pred_cls[idx]))
        if key not in seen_cluster_class:
            seen_cluster_class.add(key)
            dedup_idx.append(idx)
    dedup_idx = np.asarray(dedup_idx, dtype=int)

    selected_indices = []
    selected_labels = []
    selected_weights = []

    target_pseudo_class_mass = (pseudo_mass_share / (1.0 - pseudo_mass_share)) * n_train_per_class

    for c in range(K):
        c_mask = pred_cls[dedup_idx] == c
        c_rows = dedup_idx[c_mask][:max_per_class_quota]
        if len(c_rows) > 0:
            raw_w = margin[c_rows] - min_top2_margin + 0.10
            norm_w = raw_w / np.sum(raw_w) * target_pseudo_class_mass
            selected_indices.extend(c_rows.tolist())
            selected_labels.extend([c] * len(c_rows))
            selected_weights.extend(norm_w.tolist())

    return (
        np.asarray(selected_indices, dtype=int),
        np.asarray(selected_labels, dtype=int),
        np.asarray(selected_weights, dtype=np.float32)
    )
