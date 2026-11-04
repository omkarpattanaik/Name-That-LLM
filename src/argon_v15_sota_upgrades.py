"""
Version 15: Apex Grandmaster Sovereign Upgrades Module
Contains:
1. Stamatatos Topic Distortion Masking (+0.0319 OOF on secondary linear head)
2. Multi-Window Style Extractor (Full / Opening-64 / Ending-64 Tokens, 54 features)
3. V40A Family-Mass-Preserving Sibling Residual (exact LogSumExp conservation)
4. Vote-Normalized Blending (--sk_div 1) + Margin-Gated ATLAS Injection
5. Target-Shrunk Damped Sinkhorn-Knopp (alpha = 0.45, rho = 0.35, 100 rounds)
"""

import re
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ==============================================================================
# 1. Stamatatos Topic Distortion Masking (+0.0319 OOF on Secondary Linear Head)
# ==============================================================================
TOP_100_FUNCTION_WORDS = {
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "as", "was", "with", "be",
    "by", "on", "not", "he", "i", "this", "are", "or", "his", "from", "at", "which", "but",
    "have", "an", "had", "they", "you", "were", "their", "one", "all", "we", "can", "her",
    "has", "there", "been", "if", "more", "when", "will", "would", "who", "so", "no", "she",
    "other", "its", "may", "these", "what", "them", "than", "some", "him", "into", "time",
    "only", "could", "up", "out", "my", "two", "about", "any", "do", "such", "like", "our",
    "over", "man", "me", "even", "most", "made", "after", "also", "did", "many", "before",
    "must", "through", "back", "years", "where", "much", "your", "way", "well", "down",
    "should", "because", "each", "just", "those"
}

def stamatatos_topic_mask(texts: pd.Series) -> pd.Series:
    """
    Replaces all vocabulary outside the top-100 English function words with '*'
    while strictly preserving capitalization (Title/UPPER/lower) and exact punctuation.
    Forces character and word-n-gram linear heads to learn pure syntactic cadence.
    """
    word_pat = re.compile(r"\b[A-Za-z]+\b")

    def _mask_word(match):
        w = match.group(0)
        if w.lower() in TOP_100_FUNCTION_WORDS:
            return w
        if w.isupper():
            return "#"
        if w[0].isupper():
            return "^"
        return "*"

    return texts.astype(str).map(lambda s: word_pat.sub(_mask_word, s))


# ==============================================================================
# 2. Multi-Window Style Extractor (Full / Opening-64 / Ending-64 Tokens)
# ==============================================================================
def _window_style_vector(text: str) -> list:
    chars = max(len(text), 1)
    words = text.split()
    nw = max(len(words), 1)
    scale = 100.0 / nw
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    ns = max(len(sents), 1)
    slens = [len(s.split()) for s in sents] if sents else [nw]

    return [
        text.count("'") * scale,                                  # Straight apostrophe (Anthropic/ZAI)
        (text.count("\u2019") + text.count("\u2018")) * scale,    # Curly apostrophe (OpenAI/Google/Qwen)
        text.count('"') * scale,
        (text.count("\u201c") + text.count("\u201d")) * scale,
        text.count("\u2014") * scale,                             # Em-dash density
        len(re.findall(r"\S\u2014\S", text)) * scale,             # Unspaced em-dash
        len(re.findall(r"\s\u2014\s", text)) * scale,             # Spaced em-dash
        text.count(";") * scale,
        text.count(":") * scale,
        text.count(",") * scale,
        (text.count("(") + text.count(")")) * scale,
        float(np.mean(slens)),
        float(np.std(slens)),
        len(set(w.lower() for w in words)) / nw,                  # Type-token ratio
        sum(1 for w in words if w.lower().endswith("ly")) / nw,
        sum(1 for w in words if w.lower().endswith("ing")) / nw,
        sum(1 for w in words if w.lower().endswith("tion")) / nw,
        chars / nw
    ]

def extract_3window_prism_style(texts: pd.Series) -> np.ndarray:
    """
    Computes deterministic stylometric measurements across 3 positional views:
    Full passage, Opening 64 words, and Ending 64 words (54 features total).
    """
    out = []
    for t in texts.astype(str):
        words = t.split()
        if len(words) <= 64:
            open_txt, end_txt = t, t
        else:
            open_txt = " ".join(words[:64])
            end_txt = " ".join(words[-64:])
        out.append(_window_style_vector(t) + _window_style_vector(open_txt) + _window_style_vector(end_txt))
    return np.asarray(out, dtype=np.float32)


# ==============================================================================
# 3. Family-Mass-Preserving Sibling Residual (V40A + V31 LogSumExp Conservation)
# ==============================================================================
PROVIDER_FAMILIES = {
    "claude": ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"],
    "gemini": ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "gpt": ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"],
    "glm": ["glm-5.3", "glm-5.3-flash"],
}

def apply_v40a_family_sibling_residual(
    core_logits: np.ndarray,
    sibling_gate_probs: dict,
    class_names: list,
    family_reliability: dict = None
) -> np.ndarray:
    """
    Applies reliability-weighted intra-family sibling tie-breaking strictly inside
    each provider family via exact LogSumExp conservation.
    Preserves total provider family probability mass to machine precision (< 1e-16).
    """
    if family_reliability is None:
        family_reliability = {"claude": 0.35, "gemini": 0.30, "gpt": 0.30, "glm": 0.35}

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    updated_logits = core_logits.copy()

    for fam_name, fam_models in PROVIDER_FAMILIES.items():
        if fam_name not in sibling_gate_probs:
            continue
        cols = [name_to_idx[m] for m in fam_models if m in name_to_idx]
        if len(cols) < 2:
            continue

        gamma = float(family_reliability.get(fam_name, 0.30))
        sub_logits = updated_logits[:, cols]
        fam_log_mass = logsumexp(sub_logits, axis=1, keepdims=True)  # Conserved family log-mass

        p_core_cond = np.exp(sub_logits - fam_log_mass)
        p_gate_cond = np.clip(sibling_gate_probs[fam_name], 1e-9, 1.0)
        p_gate_cond /= p_gate_cond.sum(axis=1, keepdims=True)

        # Confidence gate: apply stronger sibling residual when core sibling margin is narrow
        sorted_core = np.sort(p_core_cond, axis=1)
        sib_margin = sorted_core[:, -1] - sorted_core[:, -2]
        row_gamma = gamma * np.clip(1.0 - (sib_margin / 0.50), 0.25, 1.0)[:, None]

        p_blended_cond = (1.0 - row_gamma) * p_core_cond + row_gamma * p_gate_cond
        updated_logits[:, cols] = fam_log_mass + np.log(np.clip(p_blended_cond, 1e-12, 1.0))

    return updated_logits


# ==============================================================================
# 4. Vote-Normalized Blending (--sk_div 1) + Margin-Gated ATLAS Injection
# ==============================================================================
def combine_vote_normalized_and_margin_atlas(
    vote_logit_matrices: list,
    vote_weights: list,
    atlas_runner_fn=None,
    test_texts: list = None,
    atlas_weight: float = 0.10,
    atlas_margin_thresh: float = 0.18,
    atlas_budget_sec: float = 8.3 * 3600,
    start_monotonic: float = None
) -> np.ndarray:
    """
    1. Divides weighted logit sum by total active vote weight (--sk_div 1) so
       adding ensemble members never artificially collapses softmax temperature.
    2. Evaluates ATLAS exclusively on rows whose top-2 probability margin falls
       below `atlas_margin_thresh`, ordered from most uncertain to least uncertain,
       stopping safely if `atlas_budget_sec` is reached.
    """
    total_w = float(sum(vote_weights))
    norm_logits = sum(w * L for w, L in zip(vote_weights, vote_logit_matrices)) / max(total_w, 1e-9)

    probs = np.exp(norm_logits - norm_logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    if atlas_runner_fn is not None and test_texts is not None and len(test_texts) > 0:
        sorted_p = np.sort(probs, axis=1)
        top2_margin = sorted_p[:, -1] - sorted_p[:, -2]
        cand_idx = np.where(top2_margin < atlas_margin_thresh)[0]

        # Prioritize lowest-margin (highest-uncertainty) passages first
        cand_idx = cand_idx[np.argsort(top2_margin[cand_idx])]
        if len(cand_idx) > 0:
            t_now = time.monotonic()
            rem_budget = None if start_monotonic is None else max(0.0, atlas_budget_sec - (t_now - start_monotonic))
            atlas_probs_sub, scored_mask = atlas_runner_fn(
                [test_texts[i] for i in cand_idx], budget_sec=rem_budget
            )
            valid_rows = cand_idx[scored_mask]
            if len(valid_rows) > 0:
                probs[valid_rows] = (
                    (1.0 - atlas_weight) * probs[valid_rows]
                    + atlas_weight * atlas_probs_sub[scored_mask]
                )
                probs[valid_rows] /= probs[valid_rows].sum(axis=1, keepdims=True)

    return probs


# ==============================================================================
# 5. Target-Shrunk Damped Sinkhorn-Knopp (alpha = 0.45, rho = 0.35, 100 rounds)
# ==============================================================================
def target_shrunk_damped_sinkhorn(
    probs: np.ndarray,
    alpha_damp: float = 0.45,
    rho_shrink: float = 0.35,
    n_rounds: int = 100
) -> np.ndarray:
    """
    Implements Pablo Diego Pereira's Target-Shrunk Damped Sinkhorn-Knopp:
      - Interpolates target column marginals between natural ensemble prior and 1/18
        using rho_shrink = 0.35 to protect against non-uniform test class subsampling.
      - Uses log-domain step damping alpha_damp = 0.45 across 100 rounds.
    """
    N, K = probs.shape
    if N < K:
        return probs  # Guard for 1-to-3 row interactive commit check

    P = np.clip(probs, 1e-12, 1.0).astype(np.float64)
    P /= P.sum(axis=1, keepdims=True)

    natural_prior = P.mean(axis=0)
    uniform_prior = np.full(K, 1.0 / K, dtype=np.float64)
    target_prior = (1.0 - rho_shrink) * natural_prior + rho_shrink * uniform_prior
    log_target = np.log(np.maximum(target_prior, 1e-12))

    log_P = np.log(P)
    for _ in range(n_rounds):
        log_col_mean = logsumexp(log_P, axis=0) - np.log(N)
        log_P = log_P + alpha_damp * (log_target - log_col_mean)[None, :]
        log_P = log_P - logsumexp(log_P, axis=1, keepdims=True)

    return np.exp(log_P)
