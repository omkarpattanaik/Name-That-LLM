"""
Data Ingestion, Clean Pool Curation, and Prompt-Paired Multi-Model Filtering.
Enforces strict host formatting, contamination purging, prompt diversity filtering,
and StratifiedGroupKFold prompt isolation.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold


CANDIDATE_18_MODELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]

CONTAMINATION_FLAGS = [
    "flag_gemini37_routing_risk",
    "flag_near_train_prompt",
    "flag_train_prompt_regen",
]


class DataCurator:
    """
    Cleans, verifies, and curates prompt-paired text datasets for LLM attribution.
    Enforces host passage constraints, de-duplication, and prompt-model balancing.
    """

    def __init__(
        self,
        candidate_models: Optional[List[str]] = None,
        min_words: int = 40,
        max_words: int = 250,
        max_tfidf_cosine: float = 0.30,
    ):
        self.candidate_models = candidate_models or CANDIDATE_18_MODELS
        self.min_words = min_words
        self.max_words = max_words
        self.max_tfidf_cosine = max_tfidf_cosine

    def purge_open_pool_contamination(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Immediately purge all rows in the Open Pool where any contamination flag is True:
        flag_gemini37_routing_risk, flag_near_train_prompt, flag_train_prompt_regen.
        """
        curated = df.copy()
        if "model" in curated.columns:
            curated = curated[curated["model"].isin(self.candidate_models)]

        for flag in CONTAMINATION_FLAGS:
            if flag in curated.columns:
                n_flagged = int(curated[flag].fillna(False).astype(bool).sum())
                if n_flagged > 0:
                    curated = curated[~curated[flag].fillna(False).astype(bool)]
                    print(f"DataCurator: purged {n_flagged} rows matching contamination flag '{flag}'")

        return curated.reset_index(drop=True)

    def enforce_host_formatting(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        text_col: str = "text",
    ) -> pd.DataFrame:
        """
        Enforce strict single-paragraph host formatting on all external rows:
        1. 40 <= n_words <= 250
        2. zero newline characters (\n)
        3. zero markdown headers (#)
        4. TF-IDF cosine similarity < max_tfidf_cosine against train.csv
        """
        initial_len = len(df)
        res = df.copy()
        res[text_col] = res[text_col].fillna("").astype(str)

        # 1. Zero newlines & zero markdown headers
        mask_clean_structure = (
            (~res[text_col].str.contains("\n", regex=False)) &
            (~res[text_col].str.contains("#", regex=False))
        )
        res = res[mask_clean_structure].copy()

        # 2. Single-paragraph word count bounds: 40 <= n_words <= 250
        word_counts = res[text_col].apply(lambda x: len(x.split()))
        mask_words = (word_counts >= self.min_words) & (word_counts <= self.max_words)
        res = res[mask_words].copy()

        # 3. TF-IDF cosine similarity < max_tfidf_cosine against train.csv
        if train_df is not None and len(res) > 0 and text_col in train_df.columns:
            train_texts = train_df[text_col].fillna("").astype(str).tolist()
            ext_texts = res[text_col].tolist()

            vec = TfidfVectorizer(max_features=15000, stop_words="english", token_pattern=r"(?u)\b\w+\b")
            X_train = vec.fit_transform(train_texts)
            X_ext = vec.transform(ext_texts)

            # Max cosine similarity across all train rows
            # X_ext is (N, D), X_train.T is (D, N_train)
            sim_matrix = X_ext @ X_train.T
            max_sims = np.asarray(sim_matrix.max(axis=1).toarray()).ravel()

            mask_low_sim = max_sims < self.max_tfidf_cosine
            n_leaked = int((~mask_low_sim).sum())
            if n_leaked > 0:
                print(f"DataCurator: purged {n_leaked} rows with TF-IDF cosine >= {self.max_tfidf_cosine} against train.csv")
            res = res[mask_low_sim].copy()

        # 4. De-duplicate identical text bodies
        res = res.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        print(f"DataCurator: formatting enforced: retained {len(res)}/{initial_len} passages ({len(res)/max(1, initial_len):.1%})")
        return res

    def filter_prompts_min_model_coverage(
        self,
        df: pd.DataFrame,
        prompt_col: str = "prompt_id",
        model_col: str = "model",
        min_models: int = 12,
    ) -> pd.DataFrame:
        """
        For 18-way backbone representation training, only include external prompt groups
        that have responses from at least min_models (default 12) of the 18 candidate models
        so that prompt topic can never leak into model identity.
        """
        if prompt_col not in df.columns or model_col not in df.columns:
            return df

        counts = df.groupby(prompt_col)[model_col].nunique()
        valid_prompts = set(counts[counts >= min_models].index)
        n_before = len(df)
        filtered = df[df[prompt_col].isin(valid_prompts)].reset_index(drop=True)
        print(f"DataCurator: retained {len(valid_prompts)} prompt groups with >= {min_models} models ({len(filtered)}/{n_before} rows)")
        return filtered

    @staticmethod
    def get_stratified_group_folds(
        df: pd.DataFrame,
        n_splits: int = 5,
        group_col: str = "prompt_id",
        target_col: str = "model",
        random_state: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Enforce 5-fold StratifiedGroupKFold(n_splits=5) grouped strictly by prompt_id.
        Never use random K-Fold.
        """
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []
        for train_idx, val_idx in sgkf.split(df, df[target_col], df[group_col]):
            folds.append((train_idx, val_idx))
        return folds

