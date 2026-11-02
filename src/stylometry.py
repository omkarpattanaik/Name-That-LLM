"""
Multi-View Stylometry and Typographic Forensics Engine.
Extracts deterministic surface stylometrics, character n-grams, word-punct n-grams,
and length-invariant typographic signatures.
"""

from collections import Counter
import math
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


RE_WORD = re.compile(r"\b\w+\b")
RE_SENTENCE = re.compile(r"[.!?]+(?:\s+|$)")
DEMONSTRATIVES = {"this", "these", "it", "they"}


class TypographicForensicsExtractor:
    """
    Extracts deterministic stylometric signatures normalized per 100 words
    to eliminate passage-length confounding.
    """

    def __init__(self):
        self.feature_names_: List[str] = [
            "straight_apos",
            "curly_apos",
            "straight_quote",
            "curly_quote",
            "em_dash",
            "en_dash",
            "semicolon",
            "colon",
            "parentheses",
            "sent_len_mean",
            "sent_len_std",
            "sent_len_iqr",
            "type_token_ratio",
            "sent_initial_demo",
            "suffix_ly",
            "suffix_ing",
            "suffix_tion",
        ]

    def extract_single(self, text: str) -> Dict[str, float]:
        t = str(text) if text is not None else ""
        words = RE_WORD.findall(t)
        n_words = max(1, len(words))
        factor = 100.0 / n_words
        words_l = [w.lower() for w in words]

        sents = [s.strip() for s in RE_SENTENCE.split(t) if s.strip()]
        if not sents:
            sents = [t]

        s_lens = [len(s.split()) for s in sents]
        s_mean = float(np.mean(s_lens))
        s_std = float(np.std(s_lens))
        if len(s_lens) > 1:
            q75, q25 = np.percentile(s_lens, [75, 25])
            s_iqr = float(q75 - q25)
        else:
            s_iqr = 0.0

        demo_cnt = sum(1 for s in sents if s.split() and s.split()[0].lower() in DEMONSTRATIVES)
        ly_cnt = sum(1 for w in words_l if w.endswith("ly"))
        ing_cnt = sum(1 for w in words_l if w.endswith("ing"))
        tion_cnt = sum(1 for w in words_l if w.endswith(("tion", "sion")))

        return {
            "straight_apos": t.count(chr(39)) * factor,
            "curly_apos": (t.count("\u2019") + t.count("\u2018")) * factor,
            "straight_quote": t.count(chr(34)) * factor,
            "curly_quote": (t.count("\u201c") + t.count("\u201d")) * factor,
            "em_dash": t.count("\u2014") * factor,
            "en_dash": t.count("\u2013") * factor,
            "semicolon": t.count(";") * factor,
            "colon": t.count(":") * factor,
            "parentheses": (t.count("(") + t.count(")")) * factor,
            "sent_len_mean": s_mean,
            "sent_len_std": s_std,
            "sent_len_iqr": s_iqr,
            "type_token_ratio": len(set(words_l)) / n_words,
            "sent_initial_demo": demo_cnt * factor,
            "suffix_ly": ly_cnt * factor,
            "suffix_ing": ing_cnt * factor,
            "suffix_tion": tion_cnt * factor,
        }

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        rows = [self.extract_single(t) for t in texts]
        df = pd.DataFrame(rows)
        return df[self.feature_names_].values.astype(np.float32)


class MultiViewTfidfExtractor:
    r"""
    View A TF-IDF Feature Extractor:
    1. char n-grams 3-6 (lowercase=False, sublinear_tf=True)
    2. char_wb n-grams 3-6 (lowercase=False, sublinear_tf=True)
    3. word n-grams 1-2 with punctuation tokens r"(?u)\b\w+\b|[^\w\s]"
    """

    def __init__(
        self,
        min_df: int = 2,
        char_max_features: int = 60000,
        char_wb_max_features: int = 60000,
        word_max_features: int = 50000,
    ):
        self.char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            min_df=min_df,
            max_features=char_max_features,
            lowercase=False,
            sublinear_tf=True,
        )
        self.char_wb_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=min_df,
            max_features=char_wb_max_features,
            lowercase=False,
            sublinear_tf=True,
        )
        self.word_vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=word_max_features,
            lowercase=False,
            sublinear_tf=True,
        )
        self.typo_extractor = TypographicForensicsExtractor()

    def fit_transform(self, texts: List[str]) -> Tuple[csr_matrix, np.ndarray]:
        X_char = self.char_vec.fit_transform(texts)
        X_wb = self.char_wb_vec.fit_transform(texts)
        X_word = self.word_vec.fit_transform(texts)
        X_typo = self.typo_extractor.extract_batch(texts)

        X_tfidf = hstack([X_char, X_wb, X_word], format="csr")
        return X_tfidf, X_typo

    def transform(self, texts: List[str]) -> Tuple[csr_matrix, np.ndarray]:
        X_char = self.char_vec.transform(texts)
        X_wb = self.char_wb_vec.transform(texts)
        X_word = self.word_vec.transform(texts)
        X_typo = self.typo_extractor.extract_batch(texts)

        X_tfidf = hstack([X_char, X_wb, X_word], format="csr")
        return X_tfidf, X_typo


# Compatibility alias
StylometryExtractor = TypographicForensicsExtractor
