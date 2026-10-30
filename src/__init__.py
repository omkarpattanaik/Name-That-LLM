"""
Name That LLM - Authorship Attribution Package
"""

from .config import CANDIDATE_MODELS, MODEL_FAMILY_MAP, NUM_CLASSES
from .metrics import compute_balanced_accuracy, evaluate_predictions
from .models_hierarchical import AdaptiveFamilyGateClassifier
from .pipeline import NameThatLLMPipeline, run_full_pipeline
from .postprocessing import sinkhorn_knopp_balancing
from .stylometry import StylometryExtractor
from .tfidf_features import DualTfidfExtractor

__all__ = [
    "CANDIDATE_MODELS",
    "MODEL_FAMILY_MAP",
    "NUM_CLASSES",
    "compute_balanced_accuracy",
    "evaluate_predictions",
    "StylometryExtractor",
    "DualTfidfExtractor",
    "AdaptiveFamilyGateClassifier",
    "sinkhorn_knopp_balancing",
    "NameThatLLMPipeline",
    "run_full_pipeline",
]
