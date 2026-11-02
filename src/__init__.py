"""
Name That LLM - Authorship Attribution SOTA Package
"""

from .config import CANDIDATE_MODELS, MODEL_FAMILY_MAP, NUM_CLASSES
from .metrics import compute_balanced_accuracy, evaluate_predictions
from .models_hierarchical import AdaptiveFamilyGateClassifier
from .pipeline import NameThatLLMPipeline, run_full_pipeline
from .postprocessing import sinkhorn_knopp_balancing

from .data_curation import DataCurator, CANDIDATE_18_MODELS, CONTAMINATION_FLAGS
from .stylometry import TypographicForensicsExtractor, MultiViewTfidfExtractor, StylometryExtractor
from .transformer_pooling import IntermediateLayerPooler, TwinPromptCenteredClassifier, group_center
from .hierarchical_gates import HierarchicalGatesEnsemble, SpecialistSubFamilyGate, FAMILY_CLAUDE, FAMILY_GEMINI, FAMILY_ZAI_OPUS
from .shared_prompt_gaussian import SharedPromptGaussianDecomposer
from .cohesion_centering import CohesionAdaptiveCentering
from .sinkhorn_calibrator import target_shrunk_sinkhorn

__all__ = [
    "CANDIDATE_MODELS",
    "CANDIDATE_18_MODELS",
    "MODEL_FAMILY_MAP",
    "NUM_CLASSES",
    "CONTAMINATION_FLAGS",
    "compute_balanced_accuracy",
    "evaluate_predictions",
    "StylometryExtractor",
    "TypographicForensicsExtractor",
    "MultiViewTfidfExtractor",
    "DualTfidfExtractor",
    "AdaptiveFamilyGateClassifier",
    "sinkhorn_knopp_balancing",
    "NameThatLLMPipeline",
    "run_full_pipeline",
    "DataCurator",
    "IntermediateLayerPooler",
    "TwinPromptCenteredClassifier",
    "group_center",
    "HierarchicalGatesEnsemble",
    "SpecialistSubFamilyGate",
    "FAMILY_CLAUDE",
    "FAMILY_GEMINI",
    "FAMILY_ZAI_OPUS",
    "SharedPromptGaussianDecomposer",
    "CohesionAdaptiveCentering",
    "target_shrunk_sinkhorn",
]
