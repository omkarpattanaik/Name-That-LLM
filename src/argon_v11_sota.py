"""
Argon V11 SOTA Architecture Module.
Implements the 5 breakthrough frontiers for the 0.700–0.730+ tier:
1. Diagonal Temperature-Scaled Log-Linear Meta-Calibration + Uncertainty-Gated ATLAS Blending
2. Sibling Cliff Disambiguator (Shape View + Targeted Syntactic & Punctuation Cadence)
3. Model-Orthogonalized Mutual k-NN Prompt Cohesion Discovery
4. Soft Intra-Cluster Author Exclusivity (Belief Propagation Repulsion) + Trimmed Prior Sinkhorn
5. Domain-Fidelity Sample Weighting Across Multi-Source Pools
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Union
from scipy.sparse import hstack, csr_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


# ==============================================================================
# FRONTIER 2: Sibling Cliff Disambiguator (Shape View + Discourse & Cadence)
# ==============================================================================
def text_to_shape_view(texts: Union[pd.Series, List[str]]) -> List[str]:
    """
    Maps characters to structural shape tokens:
    Lowercase -> 'a', Uppercase -> 'A', Digits -> '0', Punctuation/Whitespace kept exact.
    Captures intra-sentence punctuation cadence & capitalization rhythm without topic leakage.
    """
    def _transform(t: str) -> str:
        s = re.sub(r'[a-z]+', lambda m: 'a' * min(len(m.group(0)), 4), str(t))
        s = re.sub(r'[A-Z]+', lambda m: 'A' * min(len(m.group(0)), 3), s)
        s = re.sub(r'[0-9]+', '0', s)
        return s

    if isinstance(texts, pd.Series):
        return texts.map(_transform).tolist()
    return [_transform(t) for t in texts]


def extract_sibling_cliff_features(
    texts: Union[pd.Series, List[str]],
    token_logprobs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extracts targeted syntactic, burstiness, and perplexity features to separate:
      1) claude-opus-4-6 vs 4-8 vs 5
      2) gpt-5.6-luna vs sol vs terra
      3) gemini-3.6-flash vs 3.7-flash vs 3.8-flash
    """
    hedges = {'might', 'could', 'perhaps', 'generally', 'typically', 'often', 'usually', 'arguably', 'may'}
    connectives = {'consequently', 'furthermore', 'moreover', 'specifically', 'therefore', 'thus', 'hence', 'notably', 'crucially', 'ultimately'}

    rows = []
    text_list = texts.tolist() if isinstance(texts, pd.Series) else list(texts)

    for idx, t in enumerate(text_list):
        t = str(t)
        words = re.findall(r"\b\w+\b", t)
        words_lower = [w.lower() for w in words]
        nw = max(len(words), 1)
        scale = 100.0 / nw

        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
        ns = max(len(sents), 1)
        sent_lens = [len(s.split()) for s in sents] if sents else [nw]

        # 1. Opus 4.6 vs 4.8 vs 5: First-sentence framing ratio & unspaced vs spaced em-dashes
        first_sent_ratio = sent_lens[0] / max(float(np.mean(sent_lens)), 1.0)
        last_sent_ratio = sent_lens[-1] / max(float(np.mean(sent_lens)), 1.0)
        unspaced_emdash = len(re.findall(r'\S\u2014\S', t)) * scale
        spaced_emdash = len(re.findall(r'\s\u2014\s', t)) * scale
        semi_to_comma = t.count(';') / max(t.count(','), 1)

        # 2. GPT-5.6 Luna vs Sol vs Terra: Lexical complexity, Latinate compounds & hedge-to-connective ratio
        mean_word_len = float(np.mean([len(w) for w in words])) if words else 4.5
        long_word_rate = sum(1 for w in words if len(w) >= 9) / nw
        hedge_rate = sum(1 for w in words_lower if w in hedges) * scale
        conn_rate = sum(1 for w in words_lower if w in connectives) * scale
        sent_initial_conn = sum(
            1 for s in sents if s.split() and s.split()[0].lower().strip(',') in connectives
        ) / ns

        # 3. Gemini 3.6 vs 3.7 vs 3.8 Flash: Answer-first lexical overlap & round-length proximity
        if len(sents) >= 2:
            first_set = set(re.findall(r"\b\w{4,}\b", sents[0].lower()))
            rest_set = set(re.findall(r"\b\w{4,}\b", " ".join(sents[1:]).lower()))
            answer_first_overlap = len(first_set & rest_set) / max(len(first_set), 1)
        else:
            answer_first_overlap = 0.0
        dist_to_round_50 = min(abs(nw - 50), abs(nw - 100), abs(nw - 150), abs(nw - 200)) / 50.0

        # 4. Token log-probability moments from base LLM (if available)
        if token_logprobs is not None and len(token_logprobs[idx]) > 2:
            lp = np.asarray(token_logprobs[idx], dtype=np.float32)
            lp_mean = float(np.mean(lp))
            lp_std = float(np.std(lp))
            lp_q90_q10 = float(np.percentile(lp, 90) - np.percentile(lp, 10))
            lp_skew = float(np.mean(((lp - lp_mean) / max(lp_std, 1e-6)) ** 3))
        else:
            lp_mean, lp_std, lp_q90_q10, lp_skew = 0.0, 0.0, 0.0, 0.0

        rows.append([
            first_sent_ratio, last_sent_ratio, unspaced_emdash, spaced_emdash, semi_to_comma,
            mean_word_len, long_word_rate, hedge_rate, conn_rate, sent_initial_conn,
            answer_first_overlap, dist_to_round_50, lp_mean, lp_std, lp_q90_q10, lp_skew
        ])
    return np.asarray(rows, dtype=np.float32)


# ==============================================================================
# FRONTIER 3: Model-Orthogonalized Mutual k-NN Prompt Cohesion Discovery
# ==============================================================================
def discover_cohesive_prompt_clusters(
    texts: Union[pd.Series, List[str]],
    Z_embed: np.ndarray,
    plain_probs: Optional[np.ndarray] = None,
    model_centroids: Optional[np.ndarray] = None,
    w_content: float = 0.65,
    knn_k: int = 18,
    mnn_penalty: float = 0.22,
    dist_thresh: float = 0.38
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recovers shared-prompt groups in unlabeled test data with high pair precision:
      1. Extracts Content-Word TF-IDF cosine distance (D_content).
      2. Projects out predicted model style centroids from Z_embed so same-model
         passages on different prompts do not falsely cluster together (D_topic).
      3. Applies Mutual k-NN (k=18) edge verification prior to Average Linkage.
      4. Computes per-cluster cohesion scores in [0, 1] to weight twin centering.
    """
    N = len(texts)
    if N < 50:
        return np.arange(N), np.zeros(N, dtype=np.float32)

    text_list = texts.tolist() if isinstance(texts, pd.Series) else list(texts)

    # 1. Content-word TF-IDF distance
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
        max_features=50000
    )
    X_content = vec.fit_transform(text_list)
    D_content = np.clip(1.0 - (X_content @ X_content.T).toarray(), 0.0, 1.0)

    # 2. Model-orthogonalized embedding distance
    Z = np.asarray(Z_embed, dtype=np.float32).copy()
    if plain_probs is not None and model_centroids is not None:
        # Subtract expected model signature so only prompt topic remains
        Z = Z - (plain_probs @ model_centroids)
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    D_topic = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, 1.0)

    D = w_content * D_content + (1.0 - w_content) * D_topic
    np.fill_diagonal(D, 0.0)

    # 3. Mutual k-NN (k=18) verification penalty to prevent false topic merges
    k_eff = min(knn_k, N - 1)
    nn_idx = np.argpartition(D + np.eye(N) * 10.0, kth=k_eff, axis=1)[:, :k_eff]
    adj = np.zeros((N, N), dtype=bool)
    rows_grid = np.arange(N)[:, None]
    adj[rows_grid, nn_idx] = True
    mutual_knn = adj & adj.T

    D_penalized = np.where(mutual_knn, D, np.minimum(1.0, D + mnn_penalty))
    np.fill_diagonal(D_penalized, 0.0)

    # 4. Hierarchical Average Linkage + Intra-cluster cohesion weighting
    Z_link = linkage(squareform(D_penalized, checks=False), method="average")
    cluster_ids = fcluster(Z_link, t=dist_thresh, criterion="distance")

    cohesion_weights = np.zeros(N, dtype=np.float32)
    for cid in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cid)[0]
        sz = len(idx)
        if 2 <= sz <= 14:
            sub_D = D[np.ix_(idx, idx)]
            mean_dist = (np.sum(sub_D) / (sz * (sz - 1)))
            cohesion = np.clip((dist_thresh - mean_dist) / max(dist_thresh, 1e-6), 0.15, 1.0)
            # Variance correction factor sqrt((sz - 1) / sz)
            cohesion_weights[idx] = float(cohesion * np.sqrt((sz - 1.0) / sz))

    return cluster_ids, cohesion_weights


# ==============================================================================
# FRONTIER 1 & 4: Diagonal Temperature Meta-Stacking + Soft Cluster Exclusivity
# ==============================================================================
class DiagonalLogLinearMetaCalibrator:
    """
    Cross-fitted log-linear pooler with view-specific temperature and L2-regularized
    class-reliability weights, followed by soft intra-cluster exclusivity and
    prior-shrunk Sinkhorn-Knopp calibration.
    """
    def __init__(self, n_classes: int = 18, l2_reg: float = 0.05):
        self.K = n_classes
        self.l2_reg = l2_reg
        self.view_weights = None

    @staticmethod
    def _to_centered_logits(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        log_p = np.log(np.clip(probs, eps, 1.0))
        return log_p - log_p.mean(axis=1, keepdims=True)

    def fit_view_temperatures(self, oof_probs_list: list, y_true: np.ndarray) -> np.ndarray:
        M = len(oof_probs_list)
        logits_list = [self._to_centered_logits(P) for P in oof_probs_list]

        def _loss(w):
            w = np.maximum(w, 0.0)
            blended = sum(w[m] * logits_list[m] for m in range(M))
            blended -= blended.max(axis=1, keepdims=True)
            P = np.exp(blended)
            P /= P.sum(axis=1, keepdims=True)
            # Class-balanced cross-entropy proxy + smooth recall surrogate
            cls_counts = np.bincount(y_true, minlength=self.K).astype(np.float64)
            inv_freq = (len(y_true) / (self.K * np.maximum(cls_counts, 1.0)))[y_true]
            nll = -np.mean(inv_freq * np.log(np.clip(P[np.arange(len(y_true)), y_true], 1e-9, 1.0)))
            reg = self.l2_reg * np.sum((w - (1.0 / M)) ** 2)
            return nll + reg

        init_w = np.full(M, 1.0 / M)
        res = minimize(_loss, init_w, method="L-BFGS-B", bounds=[(0.0, 3.0)] * M)
        self.view_weights = res.x
        return self.view_weights

    def predict_logits(
        self,
        probs_list: list,
        atlas_probs: Optional[np.ndarray] = None,
        atlas_weight: float = 0.10,
        entropy_quantile: float = 0.60
    ) -> np.ndarray:
        logits_list = [self._to_centered_logits(P) for P in probs_list]
        w = self.view_weights if self.view_weights is not None else np.full(len(probs_list), 1.0 / len(probs_list))
        L = sum(w[m] * logits_list[m] for m in range(len(probs_list)))

        # Uncertainty-gated ATLAS injection exclusively on high-entropy rows
        if atlas_probs is not None:
            P_base = np.exp(L - L.max(axis=1, keepdims=True))
            P_base /= P_base.sum(axis=1, keepdims=True)
            ent = -np.sum(P_base * np.log(np.clip(P_base, 1e-9, 1.0)), axis=1)
            thresh = np.quantile(ent, entropy_quantile) if len(ent) >= 20 else 2.0
            uncertain_mask = ent >= thresh
            if np.any(uncertain_mask):
                L_atlas = self._to_centered_logits(atlas_probs)
                L[uncertain_mask] = (1.0 - atlas_weight) * L[uncertain_mask] + atlas_weight * L_atlas[uncertain_mask]
        return L


def apply_soft_cluster_exclusivity_and_sinkhorn(
    logits: np.ndarray,
    cluster_ids: np.ndarray,
    cohesion_weights: np.ndarray,
    beta_excl: float = 0.28,
    rho_sinkhorn: float = 0.38,
    temperature: float = 0.92,
    n_iters: int = 25
) -> np.ndarray:
    """
    1. Applies soft leave-one-out author repulsion inside cohesive prompt clusters
       (since each model answers a given prompt at most once).
    2. Applies Prior-Shrunk Sinkhorn-Knopp calibration interpolating between the
       confident natural test prior and uniform 1/18 marginals.
    """
    N, K = logits.shape
    L = logits / max(temperature, 1e-3)
    P = np.exp(L - L.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)

    if N < 50:
        return P

    # Step 1: Soft Intra-Cluster Author Exclusivity (Belief Propagation Repulsion)
    L_excl = L.copy()
    for cid in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cid)[0]
        if 2 <= len(idx) <= 14 and cohesion_weights[idx[0]] > 0:
            P_sub = np.clip(P[idx], 1e-6, 1.0 - 1e-6)
            repulsion = np.log(1.0 + (P_sub / (1.0 - P_sub)))
            # Leave-one-out sum of repulsion messages from other passages in same prompt cluster
            loo_repulsion = repulsion.sum(axis=0, keepdims=True) - repulsion
            L_excl[idx] -= beta_excl * cohesion_weights[idx, None] * loo_repulsion

    P = np.exp(L_excl - L_excl.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)

    # Step 2: Prior-Shrunk Sinkhorn-Knopp with Trimmed Natural Prior Estimation
    max_conf = P.max(axis=1)
    conf_mask = max_conf >= np.quantile(max_conf, 0.25)
    natural_prior = P[conf_mask].mean(axis=0)
    natural_prior /= natural_prior.sum()

    uniform_prior = np.full(K, 1.0 / K, dtype=np.float64)
    target_col_sums = N * ((1.0 - rho_sinkhorn) * natural_prior + rho_sinkhorn * uniform_prior)

    for _ in range(n_iters):
        P *= (target_col_sums / np.maximum(P.sum(axis=0), 1e-12))[None, :]
        P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P.astype(np.float32)


# ==============================================================================
# FRONTIER 5: Domain-Fidelity Sample Weighting Across Multi-Source Pools
# ==============================================================================
def compute_domain_protected_sample_weights(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    source_tier_shares: Optional[dict] = None
) -> np.ndarray:
    """
    Computes per-row sample weights for combined [train_df; external_df] such that:
      - Every class retains exact equal total weight across all 18 models.
      - Official train.csv holds 1 - sum(source_tier_shares) of the total mass.
      - External passages are weighted by domain fidelity p(official | style_features)
        so out-of-domain length or punctuation drift is automatically suppressed.
    """
    if source_tier_shares is None:
        source_tier_shares = {
            "in_style_multi": 0.25,  # Prvsiyan + Arxitekton (>=12 models/prompt)
            "open_pool_clean": 0.15  # Clean Open Pool v10
        }

    # Estimate domain fidelity p(train | x) using length & punctuation statistics
    all_texts = pd.concat([train_df["text"], external_df["text"]], ignore_index=True)
    dom_feats = extract_sibling_cliff_features(all_texts)[:, :6]
    dom_y = np.concatenate([np.ones(len(train_df), dtype=int), np.zeros(len(external_df), dtype=int)])

    dom_clf = LogisticRegression(C=0.2, class_weight="balanced", max_iter=500)
    dom_clf.fit(dom_feats, dom_y)
    ext_probs = np.clip(dom_clf.predict_proba(dom_feats[len(train_df):])[:, 1], 0.10, 0.90)
    fidelity_q = ext_probs / (1.0 - ext_probs)  # Importance weight p(train|x)/p(ext|x)

    total_ext_share = sum(source_tier_shares.values())
    train_share = max(1.0 - total_ext_share, 0.50)

    weights_tr = np.full(len(train_df), train_share, dtype=np.float32)
    weights_ext = np.zeros(len(external_df), dtype=np.float32)

    classes = sorted(train_df["model"].unique())
    tier_col = external_df["tier"] if "tier" in external_df.columns else pd.Series(["in_style_multi"] * len(external_df))

    for c in classes:
        n_tr_c = max(int((train_df["model"] == c).sum()), 1)
        for tier_name, alpha_s in source_tier_shares.items():
            mask = (external_df["model"] == c) & (tier_col == tier_name)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                q_sub = fidelity_q[idx]
                q_norm = q_sub / np.maximum(q_sub.mean(), 1e-6)
                weights_ext[idx] = alpha_s * (n_tr_c / float(len(idx))) * q_norm

    return np.concatenate([weights_tr, weights_ext])
