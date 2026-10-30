import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import CANDIDATE_MODELS, NUM_CLASSES
from src.cross_validation import get_prompt_grouped_cv
from src.metrics import compute_balanced_accuracy, evaluate_predictions
from src.pipeline import NameThatLLMPipeline, run_full_pipeline
from src.postprocessing import sinkhorn_knopp_balancing
from src.stylometry import StylometryExtractor
from src.synthetic_data_generator import generate_synthetic_dataset
from src.tfidf_features import DualTfidfExtractor


def test_stylometry_extractor():
    texts = [
        "This is an exemplary text passage—written with nuance; furthermore, it contains **bold** and 1. numbered lists.",
        "A straightforward sentence with no special punctuation or markdown.",
    ]
    extractor = StylometryExtractor()
    X = extractor.fit_transform(texts)
    assert X.shape[0] == 2
    assert X.shape[1] >= 60
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_tfidf_extractor():
    texts = ["Sample text with specific words", "Another distinct passage with different terms"]
    extractor = DualTfidfExtractor(word_max_features=50, char_max_features=100)
    X = extractor.fit_transform(texts)
    assert X.shape[0] == 2
    assert X.shape[1] > 0


def test_grouped_cv_leakage():
    train_df, _ = generate_synthetic_dataset(n_prompts=6)
    folds = get_prompt_grouped_cv(train_df, n_splits=3)
    assert len(folds) == 3

    for trn_idx, val_idx in folds:
        trn_prompts = set(train_df.iloc[trn_idx]["prompt_id"])
        val_prompts = set(train_df.iloc[val_idx]["prompt_id"])
        assert len(trn_prompts.intersection(val_prompts)) == 0


def test_sinkhorn_balancing():
    np.random.seed(42)
    # 50 samples, 18 classes
    raw_logits = np.random.randn(50, NUM_CLASSES)
    # Unbalanced probabilities
    exp_logits = np.exp(raw_logits)
    raw_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    balanced_probs = sinkhorn_knopp_balancing(raw_probs, max_iter=30)
    assert balanced_probs.shape == (50, NUM_CLASSES)
    # Row sums should equal 1
    np.testing.assert_allclose(np.sum(balanced_probs, axis=1), 1.0, atol=1e-3)
    # Expected column sums should be much more uniform than raw
    raw_col_std = np.std(np.mean(raw_probs, axis=0))
    balanced_col_std = np.std(np.mean(balanced_probs, axis=0))
    assert balanced_col_std <= raw_col_std + 1e-4


def test_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    score = compute_balanced_accuracy(y_true, y_pred)
    assert 0.0 <= score <= 1.0


def test_end_to_end_pipeline(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out_parquet = tmp_path / "test_submission.parquet"
    train_df, test_df = generate_synthetic_dataset(n_prompts=6)

    pipeline = NameThatLLMPipeline(n_folds=2, apply_sinkhorn=True)
    X_train, X_test = pipeline.prepare_features(train_df["text"], test_df["text"])
    eval_res = pipeline.run_cv(train_df, X_train)
    assert eval_res["balanced_accuracy"] > 0.0

    sub_df = pipeline.predict_test(X_test, test_df)
    sub_df.to_parquet(out_parquet, index=False)

    assert out_parquet.exists()
    loaded_df = pd.read_parquet(out_parquet)
    assert list(loaded_df.columns) == ["id", "model"]
    assert len(loaded_df) == len(test_df)
    assert all(m in CANDIDATE_MODELS for m in loaded_df["model"])


if __name__ == "__main__":
    print("[*] Running all tests directly...")
    test_stylometry_extractor()
    print("  [PASS] test_stylometry_extractor")
    test_tfidf_extractor()
    print("  [PASS] test_tfidf_extractor")
    test_grouped_cv_leakage()
    print("  [PASS] test_grouped_cv_leakage")
    test_sinkhorn_balancing()
    print("  [PASS] test_sinkhorn_balancing")
    test_metrics()
    print("  [PASS] test_metrics")
    from pathlib import Path
    test_end_to_end_pipeline(Path("./tmp_test_dir"))
    if Path("./tmp_test_dir/test_submission.parquet").exists():
        Path("./tmp_test_dir/test_submission.parquet").unlink()
    if Path("./tmp_test_dir").exists():
        Path("./tmp_test_dir").rmdir()
    print("  [PASS] test_end_to_end_pipeline")
    print("\n[+] ALL TESTS PASSED SUCCESSFULLY!")
