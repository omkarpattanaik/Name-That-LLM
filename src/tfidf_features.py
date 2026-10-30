"""
TF-IDF feature extraction module for Kaggle 'Name That LLM'.
Combines sublinear word n-grams and character n-grams to capture
distinctive lexical and morphological patterns.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class DualTfidfExtractor:
    """
    Extracts concatenated Word (1-2 gram) and Character (2-5 gram)
    TF-IDF features with sublinear term-frequency scaling.
    """

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_max_features: int = 15000,
        word_min_df: int = 2,
        char_ngram_range: Tuple[int, int] = (2, 5),
        char_max_features: int = 25000,
        char_min_df: int = 3,
        sublinear_tf: bool = True,
    ):
        self.word_vec = TfidfVectorizer(
            ngram_range=word_ngram_range,
            max_features=word_max_features,
            min_df=word_min_df,
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",  # Preserves punctuation tokens!
            lowercase=True,
        )

        self.char_vec = TfidfVectorizer(
            ngram_range=char_ngram_range,
            analyzer="char_wb",  # Character n-grams within word boundaries
            max_features=char_max_features,
            min_df=char_min_df,
            sublinear_tf=sublinear_tf,
            lowercase=False,  # Retains casing information!
        )

    def fit(self, texts: Union[List[str], np.ndarray]):
        texts_list = [str(t) for t in texts]
        n_docs = len(texts_list)
        if n_docs > 0:
            if self.word_vec.min_df > n_docs:
                self.word_vec.min_df = 1
            if self.char_vec.min_df > n_docs:
                self.char_vec.min_df = 1
        self.word_vec.fit(texts_list)
        self.char_vec.fit(texts_list)
        return self

    def transform(self, texts: Union[List[str], np.ndarray]) -> sp.csr_matrix:
        texts_list = [str(t) for t in texts]
        X_word = self.word_vec.transform(texts_list)
        X_char = self.char_vec.transform(texts_list)
        return sp.hstack([X_word, X_char], format="csr")

    def fit_transform(self, texts: Union[List[str], np.ndarray]) -> sp.csr_matrix:
        self.fit(texts)
        return self.transform(texts)
