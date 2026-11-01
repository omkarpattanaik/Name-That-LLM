#!/usr/bin/env python3
"""
=============================================================================
Kaggle Competition: Name That LLM  –  VERSION 2  (Elite Pipeline)
Task: Fine-grained Authorship Attribution across 18 Frontier LLMs
Evaluation Metric: Balanced Accuracy
Target File: submission.parquet [columns: 'id', 'model']
=============================================================================

ARCHITECTURE OVERVIEW:
  1. 120+ Advanced Stylometric Features (typographic fingerprints)
  2. Dual TF-IDF: char n-grams (1-6) + word n-grams (1-3) with 800K features
  3. Multi-model ensemble:
     - Logistic Regression (saga solver, multinomial)
     - Ridge Classifier
     - LinearSVC (calibrated via CalibratedClassifierCV)
     - LightGBM (if available)
     - CatBoost (if available)
  4. Prompt-Centering: cluster test passages by prompt similarity,
     then subtract cluster-mean log-probs to neutralize topic bias
  5. Sinkhorn-Knopp marginal calibration for balanced predictions
  6. StratifiedGroupKFold(5) on prompt_id for honest CV

This notebook is self-contained and runs on Kaggle (CPU/GPU, no internet).
=============================================================================
"""

import collections
import gc
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# =============================================================================
# 1.  18 Candidate Frontier Models & Family Taxonomy
# =============================================================================
CANDIDATE_MODELS: List[str] = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3", "glm-5.3-flash",
    "kimi-k3",
]
NUM_CLASSES = len(CANDIDATE_MODELS)  # 18
MODEL_TO_IDX = {m: i for i, m in enumerate(CANDIDATE_MODELS)}
IDX_TO_MODEL = {i: m for i, m in enumerate(CANDIDATE_MODELS)}

MODEL_FAMILY_MAP = {
    "gpt-5.5": "OpenAI", "gpt-5.6-luna": "OpenAI", "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI", "gpt-6-astra": "OpenAI",
    "gemini-3.1-pro-preview": "Google", "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google", "gemini-3.8-flash": "Google",
    "claude-fable-5": "Anthropic", "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic", "claude-opus-5": "Anthropic", "claude-sonnet-5": "Anthropic",
    "qwen3.8-max": "Qwen", "glm-5.3": "ZAI", "glm-5.3-flash": "ZAI", "kimi-k3": "Moonshot",
}

FAMILY_NAMES = ["OpenAI", "Anthropic", "Google", "ZAI", "Qwen", "Moonshot"]
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_NAMES)}
FAMILY_TO_MODEL_INDICES = {
    f: [MODEL_TO_IDX[m] for m, fam in MODEL_FAMILY_MAP.items() if fam == f]
    for f in FAMILY_NAMES
}

# =============================================================================
# 2.  Advanced Stylometric Feature Extractor (150+ features)
# =============================================================================
RE_SENTENCE = re.compile(r"[.!?]+(?:\s+|$)")
RE_WORD = re.compile(r"\b[a-zA-Z0-9_'-]+\b")
RE_OXFORD_COMMA = re.compile(r",\s+(?:and|or|nor|but)\b", re.IGNORECASE)
RE_NUMBERED_LIST = re.compile(r"(?:^|\n)\s*\d+[.)]\s+")
RE_BULLET_LIST = re.compile(r"(?:^|\n)\s*[-*•+]\s+")
RE_HEADER = re.compile(r"(?:^|\n)\s*#{1,6}\s+")
RE_BOLD = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
RE_ITALIC = re.compile(r"(?<!\*)\*[^*]+\*(?!\*)|(?<!_)_[^_]+_(?!_)")
RE_INLINE_CODE = re.compile(r"`[^`]+`")
RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
RE_CONTRACTION = re.compile(
    r"\b(?:don't|can't|it's|you're|we've|i'm|doesn't|won't|isn't|aren't|"
    r"haven't|hadn't|couldn't|shouldn't|wouldn't|that's|there's|what's|"
    r"who's|let's)\b", re.IGNORECASE,
)
RE_EXPANDED_FORM = re.compile(
    r"\b(?:do not|cannot|can not|it is|you are|we have|i am|does not|"
    r"will not|is not|are not|have not|had not|could not|should not|"
    r"would not|that is|there is|what is|who is|let us)\b", re.IGNORECASE,
)
RE_PARENTHETICAL = re.compile(r"\([^)]+\)")
RE_DASH_SEPARATED = re.compile(r"\s[-–—]\s")
RE_URL = re.compile(r"https?://\S+")
RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
RE_EMOJI = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001f900-\U0001f9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
    r"\U0000FE00-\U0000FE0F\U0000200D]+",
)

DISCOURSE_MARKERS = [
    "furthermore", "moreover", "in conclusion", "it is worth noting",
    "it is important to note", "crucially", "delve", "tapestry",
    "indeed", "however", "consequently", "specifically", "additionally",
    "notably", "essentially", "ultimately", "let's explore",
    "to understand this", "importantly", "in summary", "in essence",
    "on the other hand", "at its core", "step-by-step", "first and foremost",
    "similarly", "conversely", "nevertheless", "overall", "in contrast",
    "to illustrate", "for instance", "for example", "in particular",
    "as such", "as a result", "in other words", "to put it simply",
    "that said", "with that in mind", "accordingly", "hence", "thus",
    "therefore", "needless to say", "as mentioned", "to summarize",
]

PRONOUNS_1ST = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
PRONOUNS_2ND = {"you", "your", "yours", "yourself", "yourselves"}
PRONOUNS_3RD = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}
BE_VERBS = {"is", "am", "are", "was", "were", "be", "being", "been"}
MODAL_VERBS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
HEDGING_WORDS = {"perhaps", "maybe", "possibly", "likely", "unlikely", "somewhat",
                 "approximately", "roughly", "essentially", "virtually", "generally",
                 "typically", "often", "sometimes", "occasionally", "tend", "tends"}
CERTAINTY_WORDS = {"certainly", "definitely", "absolutely", "clearly", "obviously",
                   "undoubtedly", "inevitably", "necessarily", "fundamentally",
                   "unquestionably", "indisputably", "precisely"}

# Transition words that differ between model families
TRANSITION_GROUPS = {
    "additive": ["additionally", "furthermore", "moreover", "also", "besides", "likewise"],
    "adversative": ["however", "nevertheless", "nonetheless", "conversely", "yet", "still"],
    "causal": ["therefore", "consequently", "hence", "thus", "accordingly", "as a result"],
    "sequential": ["firstly", "secondly", "thirdly", "finally", "subsequently", "meanwhile"],
}


class StylometryExtractorV2:
    """Extracts 150+ specialized stylistic and typographic fingerprints.

    Improvements over V1:
    - More granular punctuation analysis (curly vs straight quotes)
    - Transition word group ratios
    - Hedging vs certainty ratios
    - Sentence-start pattern analysis
    - Information density metrics
    - More robust text statistics
    """

    def __init__(self):
        self.feature_names_: List[str] = []

    def extract_single(self, text: str) -> Dict[str, float]:
        if not isinstance(text, str) or not text.strip():
            text = " "

        feats: Dict[str, float] = {}
        text_len = len(text)
        text_lower = text.lower()

        words = RE_WORD.findall(text)
        word_count = len(words)
        words_lower = [w.lower() for w in words]
        safe_words = max(1, word_count)

        sentences = [s.strip() for s in RE_SENTENCE.split(text) if s.strip()]
        sentence_count = max(1, len(sentences))

        lines = text.split("\n")
        line_count = len(lines)
        blank_line_count = sum(1 for line in lines if not line.strip())
        non_blank_lines = [l for l in lines if l.strip()]
        paragraph_count = max(1, len(non_blank_lines))

        # === Basic text statistics ===
        feats["char_count"] = float(text_len)
        feats["word_count"] = float(word_count)
        feats["sentence_count"] = float(sentence_count)
        feats["line_count"] = float(line_count)
        feats["paragraph_count"] = float(paragraph_count)
        feats["blank_line_ratio"] = blank_line_count / max(1, line_count)
        feats["chars_per_word"] = text_len / safe_words
        feats["words_per_sentence"] = word_count / sentence_count
        feats["sentences_per_paragraph"] = sentence_count / paragraph_count

        # === Word length statistics ===
        wlens = [len(w) for w in words] if words else [0]
        feats["avg_word_length"] = float(np.mean(wlens))
        feats["std_word_length"] = float(np.std(wlens))
        feats["median_word_length"] = float(np.median(wlens))
        feats["max_word_length"] = float(np.max(wlens))
        feats["pct_long_words"] = sum(1 for l in wlens if l > 8) / safe_words
        feats["pct_short_words"] = sum(1 for l in wlens if l <= 3) / safe_words

        # === Sentence length statistics ===
        sent_lens = [len(s.split()) for s in sentences]
        feats["avg_sentence_length_words"] = float(np.mean(sent_lens)) if sent_lens else 0.0
        feats["std_sentence_length_words"] = float(np.std(sent_lens)) if sent_lens else 0.0
        feats["max_sentence_length_words"] = float(np.max(sent_lens)) if sent_lens else 0.0
        feats["min_sentence_length_words"] = float(np.min(sent_lens)) if sent_lens else 0.0
        feats["median_sentence_length"] = float(np.median(sent_lens)) if sent_lens else 0.0
        # Sentence length variance is a strong family discriminator
        feats["sentence_length_range"] = feats["max_sentence_length_words"] - feats["min_sentence_length_words"]
        if len(sent_lens) > 1:
            feats["sentence_length_cv"] = float(np.std(sent_lens) / max(1e-8, np.mean(sent_lens)))
        else:
            feats["sentence_length_cv"] = 0.0

        # === Paragraph statistics ===
        para_lens = [len(p.split()) for p in non_blank_lines]
        feats["avg_paragraph_length_words"] = float(np.mean(para_lens)) if para_lens else 0.0
        feats["std_paragraph_length_words"] = float(np.std(para_lens)) if para_lens else 0.0

        # === CRITICAL: Punctuation fingerprints (strongest signal!) ===
        feats["punct_comma"] = text.count(",") / safe_words
        feats["punct_period"] = text.count(".") / safe_words
        feats["punct_exclaim"] = text.count("!") / safe_words
        feats["punct_question"] = text.count("?") / safe_words
        feats["punct_semicolon"] = text.count(";") / safe_words
        feats["punct_colon"] = text.count(":") / safe_words
        feats["punct_ellipsis"] = text.count("…") / safe_words + text.count("...") / safe_words

        # Em-dash vs en-dash vs hyphen (major model fingerprint!)
        feats["punct_em_dash"] = text.count("—") / safe_words
        feats["punct_en_dash"] = text.count("–") / safe_words
        feats["punct_hyphen"] = text.count("-") / safe_words
        feats["dash_total"] = feats["punct_em_dash"] + feats["punct_en_dash"] + feats["punct_hyphen"]
        feats["em_dash_ratio"] = feats["punct_em_dash"] / max(1e-8, feats["dash_total"])

        # CRITICAL: Curly vs straight quotes (OpenAI/Google use different defaults!)
        feats["curly_double_open"] = text.count("\u201c") / safe_words  # "
        feats["curly_double_close"] = text.count("\u201d") / safe_words  # "
        feats["curly_single_open"] = text.count("\u2018") / safe_words  # '
        feats["curly_single_close"] = text.count("\u2019") / safe_words  # '
        feats["straight_double"] = text.count('"') / safe_words
        feats["straight_single"] = text.count("'") / safe_words
        feats["total_curly_quotes"] = (
            text.count("\u201c") + text.count("\u201d") + text.count("\u2018") + text.count("\u2019")
        ) / safe_words
        feats["total_straight_quotes"] = (text.count('"') + text.count("'")) / safe_words
        feats["curly_to_straight_ratio"] = feats["total_curly_quotes"] / max(1e-8, feats["total_straight_quotes"])

        # Parentheses & brackets
        feats["parenthetical_count"] = len(RE_PARENTHETICAL.findall(text)) / safe_words
        feats["bracket_count"] = (text.count("[") + text.count("]")) / safe_words

        # === Oxford comma (strong family signal) ===
        feats["oxford_comma_count"] = len(RE_OXFORD_COMMA.findall(text))
        feats["oxford_comma_ratio"] = feats["oxford_comma_count"] / sentence_count

        # === Contraction vs expanded form ===
        contraction_count = len(RE_CONTRACTION.findall(text_lower))
        expanded_count = len(RE_EXPANDED_FORM.findall(text_lower))
        feats["contraction_count"] = contraction_count
        feats["expanded_form_count"] = expanded_count
        feats["contraction_ratio"] = contraction_count / max(1, contraction_count + expanded_count)

        # === Markdown formatting ===
        feats["has_numbered_list"] = float(bool(RE_NUMBERED_LIST.search(text)))
        feats["has_bullet_list"] = float(bool(RE_BULLET_LIST.search(text)))
        feats["numbered_list_count"] = len(RE_NUMBERED_LIST.findall(text))
        feats["bullet_list_count"] = len(RE_BULLET_LIST.findall(text))
        feats["has_header"] = float(bool(RE_HEADER.search(text)))
        feats["header_count"] = len(RE_HEADER.findall(text))
        feats["bold_count"] = len(RE_BOLD.findall(text))
        feats["italic_count"] = len(RE_ITALIC.findall(text))
        feats["inline_code_count"] = len(RE_INLINE_CODE.findall(text))
        feats["code_block_count"] = len(RE_CODE_BLOCK.findall(text))
        feats["formatting_density"] = (
            feats["bold_count"] + feats["italic_count"] +
            feats["inline_code_count"] + feats["code_block_count"]
        ) / sentence_count

        # === Vocabulary richness ===
        unique_words = set(words_lower)
        feats["unique_word_ratio"] = len(unique_words) / safe_words
        feats["hapax_legomena_ratio"] = sum(
            1 for w in unique_words if words_lower.count(w) == 1
        ) / safe_words
        feats["hapax_dislegomena_ratio"] = sum(
            1 for w in unique_words if words_lower.count(w) == 2
        ) / safe_words

        # Yule's K measure of vocabulary richness
        freq_spectrum = collections.Counter(collections.Counter(words_lower).values())
        N = safe_words
        M2 = sum(i * i * freq_spectrum[i] for i in freq_spectrum)
        feats["yule_k"] = 10000 * (M2 - N) / max(1, N * N) if N > 0 else 0.0

        # === Discourse markers ===
        dm_total = 0
        for marker in DISCOURSE_MARKERS:
            c = text_lower.count(marker)
            dm_total += c
        feats["discourse_marker_density"] = dm_total / sentence_count

        # Specific high-signal markers
        feats["dm_delve"] = text_lower.count("delve") / safe_words
        feats["dm_tapestry"] = text_lower.count("tapestry") / safe_words
        feats["dm_indeed"] = text_lower.count("indeed") / safe_words
        feats["dm_furthermore"] = text_lower.count("furthermore") / safe_words
        feats["dm_moreover"] = text_lower.count("moreover") / safe_words
        feats["dm_crucially"] = text_lower.count("crucially") / safe_words
        feats["dm_notably"] = text_lower.count("notably") / safe_words
        feats["dm_essentially"] = text_lower.count("essentially") / safe_words
        feats["dm_ultimately"] = text_lower.count("ultimately") / safe_words
        feats["dm_however"] = text_lower.count("however") / safe_words

        # === Transition word groups ===
        for group_name, markers in TRANSITION_GROUPS.items():
            count = sum(text_lower.count(m) for m in markers)
            feats[f"transition_{group_name}"] = count / sentence_count

        # === Hedging vs certainty ===
        hedge_count = sum(1 for w in words_lower if w in HEDGING_WORDS)
        certainty_count = sum(1 for w in words_lower if w in CERTAINTY_WORDS)
        feats["hedging_ratio"] = hedge_count / safe_words
        feats["certainty_ratio"] = certainty_count / safe_words
        feats["hedge_to_certainty"] = hedge_count / max(1, certainty_count)

        # === Modal verbs ===
        modal_count = sum(1 for w in words_lower if w in MODAL_VERBS)
        feats["modal_verb_ratio"] = modal_count / safe_words

        # === Pronoun analysis ===
        p1 = sum(1 for w in words_lower if w in PRONOUNS_1ST)
        p2 = sum(1 for w in words_lower if w in PRONOUNS_2ND)
        p3 = sum(1 for w in words_lower if w in PRONOUNS_3RD)
        feats["pronoun_1st_ratio"] = p1 / safe_words
        feats["pronoun_2nd_ratio"] = p2 / safe_words
        feats["pronoun_3rd_ratio"] = p3 / safe_words
        feats["pronoun_total"] = (p1 + p2 + p3) / safe_words

        # === Be-verb analysis ===
        be_count = sum(1 for w in words_lower if w in BE_VERBS)
        feats["be_verb_ratio"] = be_count / safe_words

        # === Sentence-start analysis (strong discriminator!) ===
        sentence_starts = []
        for s in sentences:
            s_words = s.strip().split()
            if s_words:
                sentence_starts.append(s_words[0].lower())
        feats["start_with_the"] = sum(1 for s in sentence_starts if s == "the") / sentence_count
        feats["start_with_this"] = sum(1 for s in sentence_starts if s == "this") / sentence_count
        feats["start_with_it"] = sum(1 for s in sentence_starts if s == "it") / sentence_count
        feats["start_with_in"] = sum(1 for s in sentence_starts if s == "in") / sentence_count
        feats["start_with_for"] = sum(1 for s in sentence_starts if s == "for") / sentence_count
        feats["start_with_a"] = sum(1 for s in sentence_starts if s in ("a", "an")) / sentence_count
        feats["start_diversity"] = len(set(sentence_starts)) / sentence_count

        # === Whitespace patterns ===
        feats["double_space"] = text.count("  ") / safe_words
        feats["trailing_space_lines"] = sum(1 for l in lines if l != l.rstrip()) / max(1, line_count)

        # === Special characters ===
        feats["has_url"] = float(bool(RE_URL.search(text)))
        feats["has_email"] = float(bool(RE_EMAIL.search(text)))
        feats["emoji_count"] = len(RE_EMOJI.findall(text))
        feats["dash_separated_count"] = len(RE_DASH_SEPARATED.findall(text)) / safe_words

        # === Information density ===
        feats["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / max(1, text_len)
        feats["digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(1, text_len)
        feats["space_ratio"] = sum(1 for c in text if c == " ") / max(1, text_len)
        feats["alpha_ratio"] = sum(1 for c in text if c.isalpha()) / max(1, text_len)
        feats["non_ascii_ratio"] = sum(1 for c in text if ord(c) > 127) / max(1, text_len)

        # === Character-level diversity (entropy) ===
        char_freq = collections.Counter(text_lower)
        char_probs = np.array([v / text_len for v in char_freq.values()])
        feats["char_entropy"] = float(-np.sum(char_probs * np.log2(char_probs + 1e-12)))

        # Word-level entropy
        word_freq = collections.Counter(words_lower)
        word_probs = np.array([v / safe_words for v in word_freq.values()])
        feats["word_entropy"] = float(-np.sum(word_probs * np.log2(word_probs + 1e-12)))

        # === Readability scores ===
        total_syllables = sum(self._count_syllables(w) for w in words)
        feats["avg_syllables_per_word"] = total_syllables / safe_words
        # Flesch Reading Ease
        feats["flesch_reading_ease"] = (
            206.835
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (total_syllables / safe_words)
        )
        # Automated Readability Index
        feats["ari"] = (
            4.71 * (text_len / safe_words)
            + 0.5 * (word_count / sentence_count)
            - 21.43
        )

        # === Sentence ending patterns ===
        feats["end_period"] = sum(1 for s in sentences if s.rstrip().endswith(".")) / sentence_count
        feats["end_exclaim"] = sum(1 for s in sentences if s.rstrip().endswith("!")) / sentence_count
        feats["end_question"] = sum(1 for s in sentences if s.rstrip().endswith("?")) / sentence_count
        feats["end_colon"] = sum(1 for s in sentences if s.rstrip().endswith(":")) / sentence_count

        # === Line-level features ===
        line_lens = [len(l) for l in lines]
        feats["avg_line_length"] = float(np.mean(line_lens)) if line_lens else 0.0
        feats["std_line_length"] = float(np.std(line_lens)) if line_lens else 0.0
        feats["max_line_length"] = float(np.max(line_lens)) if line_lens else 0.0

        return feats

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Approximate syllable count."""
        word = word.lower().rstrip("e")
        vowels = "aeiou"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(1, count)

    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """Extract features for a list of texts."""
        rows = []
        for text in texts:
            rows.append(self.extract_single(text))
        df = pd.DataFrame(rows)
        self.feature_names_ = list(df.columns)
        return df


# =============================================================================
# 3.  TF-IDF Feature Builder (Massive Feature Space)
# =============================================================================
class TfidfFeatureBuilder:
    """Build powerful TF-IDF representations from text.

    Uses separate char and word n-gram vectorizers with high feature counts,
    sublinear TF, and smart tokenization patterns that capture punctuation.
    """

    def __init__(
        self,
        char_ngram_range: Tuple[int, int] = (1, 6),
        word_ngram_range: Tuple[int, int] = (1, 3),
        char_max_features: int = 500_000,
        word_max_features: int = 300_000,
    ):
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            max_features=char_max_features,
            sublinear_tf=True,
            strip_accents=None,
            min_df=2,
        )
        self.word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            max_features=word_max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            strip_accents=None,
            min_df=2,
        )
        # Additional: pure char n-grams (not word-boundary-aware)
        self.char_pure_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=200_000,
            sublinear_tf=True,
            min_df=2,
        )

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        X_char_wb = self.char_vec.fit_transform(texts)
        X_word = self.word_vec.fit_transform(texts)
        X_char_pure = self.char_pure_vec.fit_transform(texts)
        return sp.hstack([X_char_wb, X_word, X_char_pure], format="csr")

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        X_char_wb = self.char_vec.transform(texts)
        X_word = self.word_vec.transform(texts)
        X_char_pure = self.char_pure_vec.transform(texts)
        return sp.hstack([X_char_wb, X_word, X_char_pure], format="csr")


# =============================================================================
# 4.  Prompt Centering (Cluster test passages by prompt similarity)
# =============================================================================
def cluster_prompts_for_centering(
    X_tfidf: sp.csr_matrix,
    n_clusters: int = None,
    n_components: int = 50,
) -> np.ndarray:
    """Cluster text passages by their TF-IDF similarity to discover prompt groups.

    In the competition, 18 models answer the same prompts. By clustering the
    test passages into groups of ~18, we can identify which passages share
    the same prompt and apply prompt-centering.
    """
    if n_clusters is None:
        n_clusters = max(1, X_tfidf.shape[0] // 18)

    # Reduce dimensionality for clustering
    svd = TruncatedSVD(n_components=min(n_components, X_tfidf.shape[1] - 1), random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    # Normalize for cosine-like clustering
    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    X_reduced = X_reduced / np.maximum(norms, 1e-8)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward",
    )
    labels = clustering.fit_predict(X_reduced)
    return labels


def apply_prompt_centering(
    log_probs: np.ndarray,
    cluster_labels: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Apply soft prompt centering to log-probabilities.

    For each cluster (proxy for prompt group), subtract the cluster mean
    of log-probabilities. This removes topic bias and focuses on
    model-specific stylistic signals.

    Args:
        log_probs: (n_samples, n_classes) array of log-probabilities
        cluster_labels: (n_samples,) cluster assignments
        alpha: strength of centering (0=none, 1=full centering)
    """
    centered = log_probs.copy()
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        if mask.sum() < 2:
            continue
        cluster_mean = log_probs[mask].mean(axis=0, keepdims=True)
        centered[mask] = log_probs[mask] - alpha * cluster_mean
    return centered


# =============================================================================
# 5.  Sinkhorn-Knopp Marginal Calibration
# =============================================================================
def sinkhorn_calibrate(
    probs: np.ndarray,
    target_marginals: Optional[np.ndarray] = None,
    n_iter: int = 30,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply Sinkhorn-Knopp to calibrate prediction marginals.

    Ensures that the average prediction distribution matches the
    expected uniform distribution (1/18 per class).
    """
    if target_marginals is None:
        target_marginals = np.ones(probs.shape[1]) / probs.shape[1]

    P = probs.copy().astype(np.float64) + eps
    target_row = np.ones(P.shape[0]) / P.shape[0]

    for _ in range(n_iter):
        # Column normalization
        col_sums = P.sum(axis=0)
        P *= (target_marginals / (col_sums + eps))[np.newaxis, :]
        # Row normalization
        row_sums = P.sum(axis=1)
        P *= (target_row / (row_sums + eps))[:, np.newaxis]

    # Renormalize rows to valid probability distributions
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, eps)
    return P


# =============================================================================
# 6.  Multi-Model Ensemble
# =============================================================================
def build_ensemble_models():
    """Build a diverse set of classifiers for ensembling."""
    models = {}

    # 1. Logistic Regression (strong for TF-IDF)
    models["lr_saga"] = LogisticRegression(
        solver="saga",
        multi_class="multinomial",
        max_iter=2000,
        C=3.0,
        penalty="l2",
        random_state=42,
        n_jobs=-1,
    )

    # 2. Logistic Regression with stronger regularization
    models["lr_lbfgs"] = LogisticRegression(
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=2000,
        C=1.0,
        penalty="l2",
        random_state=43,
        n_jobs=-1,
    )

    # 3. Ridge Classifier
    models["ridge"] = RidgeClassifier(
        alpha=1.0,
        class_weight="balanced",
        random_state=42,
    )

    # 4. Calibrated LinearSVC
    models["svc"] = CalibratedClassifierCV(
        LinearSVC(
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
        cv=3,
        method="sigmoid",
    )

    # 5. Try LightGBM if available
    try:
        import lightgbm as lgb
        models["lgb"] = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.5,
            min_child_samples=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        pass

    return models


def get_model_probabilities(model, X, classes):
    """Get probability predictions from any sklearn-compatible model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(X)
        if decisions.ndim == 1:
            # Binary case, shouldn't happen here
            probs = np.column_stack([1 - decisions, decisions])
        else:
            probs = softmax(decisions, axis=1)
        return probs
    else:
        raise ValueError(f"Model {type(model)} has no predict_proba or decision_function")


# =============================================================================
# 7.  Main Pipeline
# =============================================================================
def find_data_dir() -> Path:
    """Find the competition data directory."""
    candidates = [
        Path("/kaggle/input/name-that-llm"),
        Path("/kaggle/input/competitions/name-that-llm"),
        Path("data"),
        Path("../input/name-that-llm"),
    ]
    for c in candidates:
        if (c / "train.csv").exists():
            return c
    raise FileNotFoundError("Cannot find competition data directory")


def run_pipeline():
    """Full V2 pipeline: feature extraction → ensemble training → prediction."""
    T0 = time.time()

    # --- Load Data ---
    DATA_DIR = find_data_dir()
    print(f"[INFO] Data directory: {DATA_DIR}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"[INFO] Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"[INFO] Models: {sorted(train_df['model'].unique())}")

    # Encode labels
    le = LabelEncoder()
    le.classes_ = np.array(CANDIDATE_MODELS)
    train_df["label"] = le.transform(train_df["model"])

    all_texts = pd.concat([train_df["text"], test_df["text"]], ignore_index=True).fillna("").tolist()
    train_texts = train_df["text"].fillna("").tolist()
    test_texts = test_df["text"].fillna("").tolist()

    # --- Step 1: Stylometric Features ---
    print(f"[{time.time()-T0:.1f}s] Extracting stylometric features...")
    extractor = StylometryExtractorV2()
    stylo_train = extractor.extract_batch(train_texts)
    stylo_test = extractor.extract_batch(test_texts)

    # Fill NaN and inf
    stylo_train = stylo_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    stylo_test = stylo_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale stylometric features
    scaler = StandardScaler()
    stylo_train_scaled = scaler.fit_transform(stylo_train.values)
    stylo_test_scaled = scaler.transform(stylo_test.values)

    print(f"  → {stylo_train.shape[1]} stylometric features extracted")

    # --- Step 2: TF-IDF Features ---
    print(f"[{time.time()-T0:.1f}s] Building TF-IDF features...")
    tfidf_builder = TfidfFeatureBuilder(
        char_ngram_range=(1, 6),
        word_ngram_range=(1, 3),
        char_max_features=500_000,
        word_max_features=300_000,
    )

    X_tfidf_train = tfidf_builder.fit_transform(train_texts)
    X_tfidf_test = tfidf_builder.transform(test_texts)

    print(f"  → TF-IDF shape: {X_tfidf_train.shape}")

    # --- Step 3: Combine features ---
    X_train_full = sp.hstack([
        X_tfidf_train,
        sp.csr_matrix(stylo_train_scaled),
    ], format="csr")

    X_test_full = sp.hstack([
        X_tfidf_test,
        sp.csr_matrix(stylo_test_scaled),
    ], format="csr")

    y_train = train_df["label"].values
    groups = train_df["prompt_id"].values

    print(f"  → Combined feature matrix: {X_train_full.shape}")

    # --- Step 4: Cross-Validation & Ensemble Training ---
    print(f"[{time.time()-T0:.1f}s] Training ensemble with StratifiedGroupKFold...")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    models_dict = build_ensemble_models()

    oof_probs = {name: np.zeros((len(train_df), NUM_CLASSES)) for name in models_dict}
    test_probs = {name: np.zeros((len(test_df), NUM_CLASSES)) for name in models_dict}
    fold_scores = {name: [] for name in models_dict}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_full, y_train, groups)):
        print(f"\n  === Fold {fold_idx+1}/5 ===")
        X_tr = X_train_full[train_idx]
        X_val = X_train_full[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        for name, model in models_dict.items():
            try:
                t0 = time.time()

                # For LightGBM, convert to dense or use special handling
                if name == "lgb":
                    # LightGBM can handle sparse directly
                    model_instance = model.__class__(**model.get_params())
                    model_instance.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            __import__("lightgbm").early_stopping(50, verbose=False),
                            __import__("lightgbm").log_evaluation(0),
                        ],
                    )
                else:
                    model_instance = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
                    # For CalibratedClassifierCV, we need special handling
                    if name == "svc":
                        model_instance = CalibratedClassifierCV(
                            LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42),
                            cv=3, method="sigmoid",
                        )
                    model_instance.fit(X_tr, y_tr)

                # Get probabilities
                probs_val = get_model_probabilities(model_instance, X_val, le.classes_)
                probs_test = get_model_probabilities(model_instance, X_test_full, le.classes_)

                oof_probs[name][val_idx] = probs_val
                test_probs[name] += probs_test / 5

                score = balanced_accuracy_score(y_val, probs_val.argmax(axis=1))
                fold_scores[name].append(score)
                print(f"    {name}: BA = {score:.4f} ({time.time()-t0:.1f}s)")

            except Exception as e:
                print(f"    {name}: FAILED - {e}")
                fold_scores[name].append(0.0)

        gc.collect()

    # --- Print CV results ---
    print(f"\n{'='*60}")
    print("  Cross-Validation Results (StratifiedGroupKFold)")
    print(f"{'='*60}")
    model_weights = {}
    for name in models_dict:
        scores = fold_scores[name]
        mean_score = np.mean(scores) if scores else 0.0
        std_score = np.std(scores) if scores else 0.0
        print(f"  {name:15s}: {mean_score:.4f} ± {std_score:.4f}")
        model_weights[name] = mean_score  # Use CV score as weight

    # --- Step 5: Weighted Ensemble ---
    print(f"\n[{time.time()-T0:.1f}s] Building weighted ensemble...")

    # Normalize weights
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v / total_weight for k, v in model_weights.items()}
    else:
        model_weights = {k: 1.0 / len(model_weights) for k in model_weights}

    # Compute OOF ensemble
    oof_ensemble = np.zeros((len(train_df), NUM_CLASSES))
    test_ensemble = np.zeros((len(test_df), NUM_CLASSES))

    for name in models_dict:
        w = model_weights[name]
        oof_ensemble += w * oof_probs[name]
        test_ensemble += w * test_probs[name]
        print(f"  {name}: weight = {w:.4f}")

    oof_ba = balanced_accuracy_score(y_train, oof_ensemble.argmax(axis=1))
    print(f"\n  OOF Ensemble BA: {oof_ba:.4f}")

    # --- Step 6: Prompt Centering on test predictions ---
    if len(test_df) > 18:
        print(f"\n[{time.time()-T0:.1f}s] Applying prompt centering...")
        test_cluster_labels = cluster_prompts_for_centering(X_tfidf_test)
        test_log_probs = np.log(test_ensemble + 1e-12)
        centered_log_probs = apply_prompt_centering(test_log_probs, test_cluster_labels, alpha=0.3)
        test_probs_final = softmax(centered_log_probs, axis=1)
    else:
        print(f"\n[{time.time()-T0:.1f}s] Test set too small for prompt centering, skipping...")
        test_probs_final = test_ensemble

    # --- Step 7: Sinkhorn calibration ---
    print(f"[{time.time()-T0:.1f}s] Applying Sinkhorn-Knopp calibration...")
    test_probs_calibrated = sinkhorn_calibrate(test_probs_final)

    # --- Step 8: Generate predictions ---
    test_preds = test_probs_calibrated.argmax(axis=1)
    test_labels = [IDX_TO_MODEL[i] for i in test_preds]

    submission = pd.DataFrame({
        "id": test_df["id"],
        "model": test_labels,
    })

    # --- Save submission ---
    out_path = Path("submission.parquet")
    submission.to_parquet(out_path, index=False)
    print(f"\n[{time.time()-T0:.1f}s] Submission saved to {out_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Predictions distribution:")
    print(submission["model"].value_counts().to_string())
    print(f"\n  Total time: {time.time()-T0:.1f}s")

    return submission


# =============================================================================
# 8.  Entry Point
# =============================================================================
if __name__ == "__main__":
    submission = run_pipeline()
    print("\n✅ V2 pipeline complete!")
