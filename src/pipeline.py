"""
Master Pipeline module for Kaggle 'Name That LLM'.
Coordinates feature extraction, prompt-grouped cross-validation, model training,
Sinkhorn post-processing, evaluation, and submission parquet generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import (
    CANDIDATE_MODELS,
    DEFAULT_KAGGLE_INPUT_PATHS,
    ID_COLUMN,
    IDX_TO_MODEL,
    MODEL_TO_IDX,
    N_FOLDS,
    PROMPT_GROUP_COLUMN,
    RANDOM_SEED,
    SUBMISSION_FILENAME,
    TARGET_COLUMN,
    TEXT_COLUMN,
)
from .cross_validation import get_prompt_grouped_cv
from .metrics import evaluate_predictions, format_evaluation_report
from .models_hierarchical import AdaptiveFamilyGateClassifier
from .models_tabular import CalibratedRidgeModel, FastSGDModel
from .postprocessing import sinkhorn_knopp_balancing, temperature_scaled_adjustment
from .stylometry import StylometryExtractor
from .synthetic_data_generator import generate_synthetic_dataset
from .tfidf_features import DualTfidfExtractor


def find_competition_data_dir() -> Optional[Path]:
    """Dynamically locates the competition data directory across common Kaggle & local paths."""
    for p in DEFAULT_KAGGLE_INPUT_PATHS:
        if (p / "train.csv").exists():
            return p
    # Search recursively in /kaggle/input if on Kaggle
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        found = list(kaggle_root.glob("**/train.csv"))
        if found:
            return found[0].parent
    return None


class NameThatLLMPipeline:
    """
    End-to-end Machine Learning Pipeline for 'Name That LLM'.
    """

    def __init__(
        self,
        n_folds: int = N_FOLDS,
        apply_sinkhorn: bool = True,
        hierarchical_weight: float = 0.4,
        random_state: int = RANDOM_SEED,
    ):
        self.n_folds = n_folds
        self.apply_sinkhorn = apply_sinkhorn
        self.hierarchical_weight = hierarchical_weight
        self.random_state = random_state

        self.stylometry_extractor = StylometryExtractor()
        self.tfidf_extractor = DualTfidfExtractor()

        self.models: List[AdaptiveFamilyGateClassifier] = []
        self.oof_predictions: Optional[np.ndarray] = None
        self.eval_results_: Optional[Dict] = None

    def prepare_features(
        self,
        train_texts: Union[List[str], pd.Series],
        test_texts: Optional[Union[List[str], pd.Series]] = None,
    ) -> Tuple[sp.csr_matrix, Optional[sp.csr_matrix]]:
        """
        Extracts stylometric and TF-IDF features and concatenates them into
        a unified sparse feature matrix.
        """
        print(f"[*] Extracting 120+ stylometric signals for {len(train_texts)} train texts...")
        X_sty_train = self.stylometry_extractor.fit_transform(train_texts)

        print("[*] Extracting sublinear word & character n-gram TF-IDF...")
        X_tfidf_train = self.tfidf_extractor.fit_transform(train_texts)

        # Scale dense stylometry features and convert to sparse
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_sty_train_scaled = sp.csr_matrix(scaler.fit_transform(X_sty_train))

        X_train_combined = sp.hstack([X_sty_train_scaled, X_tfidf_train], format="csr")

        X_test_combined = None
        if test_texts is not None and len(test_texts) > 0:
            print(f"[*] Extracting features for {len(test_texts)} test texts...")
            X_sty_test = self.stylometry_extractor.transform(test_texts)
            X_sty_test_scaled = sp.csr_matrix(scaler.transform(X_sty_test))
            X_tfidf_test = self.tfidf_extractor.transform(test_texts)
            X_test_combined = sp.hstack([X_sty_test_scaled, X_tfidf_test], format="csr")

        return X_train_combined, X_test_combined

    def run_cv(
        self,
        train_df: pd.DataFrame,
        X_train: sp.csr_matrix,
    ) -> Dict:
        """
        Runs StratifiedGroupKFold on prompt_id and trains fold models.
        """
        y_train = np.array([MODEL_TO_IDX[m] for m in train_df[TARGET_COLUMN]])
        prompt_groups = train_df[PROMPT_GROUP_COLUMN].values

        folds = get_prompt_grouped_cv(
            train_df,
            target_col=TARGET_COLUMN,
            group_col=PROMPT_GROUP_COLUMN,
            n_splits=self.n_folds,
            random_state=self.random_state,
        )

        N = len(train_df)
        K = len(CANDIDATE_MODELS)
        oof_probs = np.zeros((N, K), dtype=np.float32)
        self.models = []

        print(f"\n[*] Starting {self.n_folds}-Fold StratifiedGroupKFold on prompt_id...")
        for fold, (trn_idx, val_idx) in enumerate(folds):
            X_trn, y_trn = X_train[trn_idx], y_train[trn_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            model = AdaptiveFamilyGateClassifier(
                alpha_family=1.0,
                alpha_model=0.8,
                hierarchical_weight=self.hierarchical_weight,
            )
            model.fit(X_trn, y_trn)
            self.models.append(model)

            val_probs = model.predict_proba(X_val)
            oof_probs[val_idx] = val_probs

            val_pred = np.argmax(val_probs, axis=1)
            from sklearn.metrics import balanced_accuracy_score
            fold_bal_acc = balanced_accuracy_score(y_val, val_pred)
            print(f"  -> Fold {fold + 1}/{self.n_folds} Balanced Accuracy: {fold_bal_acc:.4f}")

        # Out-of-fold evaluation (Raw)
        self.oof_predictions = oof_probs
        oof_preds_raw = np.argmax(oof_probs, axis=1)
        raw_eval = evaluate_predictions(y_train, oof_preds_raw)
        print("\n" + "=" * 60)
        print("RAW OUT-OF-FOLD EVALUATION")
        print("=" * 60)
        print(format_evaluation_report(raw_eval))

        # Sinkhorn-Knopp Balanced Out-of-fold evaluation
        if self.apply_sinkhorn:
            oof_probs_sinkhorn = sinkhorn_knopp_balancing(oof_probs)
            oof_preds_sinkhorn = np.argmax(oof_probs_sinkhorn, axis=1)
            sinkhorn_eval = evaluate_predictions(y_train, oof_preds_sinkhorn)
            print("\n" + "=" * 60)
            print("SINKHORN-KNOPP CALIBRATED OUT-OF-FOLD EVALUATION")
            print("=" * 60)
            print(format_evaluation_report(sinkhorn_eval))
            self.eval_results_ = sinkhorn_eval
        else:
            self.eval_results_ = raw_eval

        return self.eval_results_

    def predict_test(
        self,
        X_test: sp.csr_matrix,
        test_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Averages predictions across all fold models, applies Sinkhorn balancing,
        and returns a submission DataFrame.
        """
        N_test = X_test.shape[0]
        K = len(CANDIDATE_MODELS)
        test_probs = np.zeros((N_test, K), dtype=np.float32)

        for model in self.models:
            test_probs += model.predict_proba(X_test) / len(self.models)

        if self.apply_sinkhorn:
            test_probs = sinkhorn_knopp_balancing(test_probs)

        pred_indices = np.argmax(test_probs, axis=1)
        pred_models = [IDX_TO_MODEL[i] for i in pred_indices]

        if test_df is not None and ID_COLUMN in test_df.columns:
            ids = test_df[ID_COLUMN].values
        else:
            ids = [f"test_{i}" for i in range(N_test)]

        sub_df = pd.DataFrame({
            ID_COLUMN: ids,
            TARGET_COLUMN: pred_models,
        })
        return sub_df


def run_full_pipeline(
    data_dir: Optional[Union[str, Path]] = None,
    output_path: str = SUBMISSION_FILENAME,
    n_folds: int = N_FOLDS,
    use_synthetic_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Executes the complete pipeline: data discovery, CV, model fitting, and submission.
    """
    if data_dir is None:
        found_dir = find_competition_data_dir()
    else:
        found_dir = Path(data_dir)

    if found_dir is not None and (found_dir / "train.csv").exists():
        print(f"[*] Found competition data in: {found_dir.resolve()}")
        train_df = pd.read_csv(found_dir / "train.csv")
        test_path = found_dir / "test.csv"
        test_df = pd.read_csv(test_path) if test_path.exists() else None
    elif use_synthetic_if_missing:
        print("[!] No competition data found on disk. Generating synthetic data conforming to competition schema...")
        train_df, test_df = generate_synthetic_dataset(n_prompts=15)
        print(f"[*] Generated synthetic train: {len(train_df)} rows, test: {len(test_df)} rows.")
    else:
        raise FileNotFoundError("Could not find competition data directory.")

    pipeline = NameThatLLMPipeline(n_folds=n_folds, apply_sinkhorn=True)
    test_texts = test_df[TEXT_COLUMN] if test_df is not None else None
    X_train, X_test = pipeline.prepare_features(train_df[TEXT_COLUMN], test_texts)

    pipeline.run_cv(train_df, X_train)

    if X_test is not None:
        sub_df = pipeline.predict_test(X_test, test_df)
        sub_df.to_parquet(output_path, index=False)
        print(f"\n[+] Successfully generated submission parquet: {output_path}")
        print(f"    Total rows: {len(sub_df)}")
        print(f"    Sample head:\n{sub_df.head(5)}")
        return sub_df

    return pd.DataFrame()
