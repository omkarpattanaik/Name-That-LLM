"""
Cross-Validation module for Kaggle 'Name That LLM'.
Implements strict Group-aware Stratification to prevent topic leakage.
"""

from typing import Generator, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import N_FOLDS, RANDOM_SEED


def get_prompt_grouped_cv(
    df: pd.DataFrame,
    target_col: str = "model",
    group_col: str = "prompt_id",
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Constructs train/validation fold indices using StratifiedGroupKFold.
    Ensures that ALL responses to a given prompt remain exclusively in
    EITHER the training fold OR the validation fold, preventing topic leakage.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    X = np.zeros(len(df))
    y = df[target_col].values
    groups = df[group_col].values

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        # Verify zero group leakage
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        leakage = train_groups.intersection(val_groups)
        assert len(leakage) == 0, f"Critical Data Leakage: Fold {fold} has {len(leakage)} overlapping prompt_ids!"

        folds.append((train_idx, val_idx))

    return folds
