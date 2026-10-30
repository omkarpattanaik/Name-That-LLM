"""
Configuration and constants for Kaggle 'Name That LLM' competition.
"""

from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# The 18 Candidate Frontier LLMs
# ---------------------------------------------------------------------------
CANDIDATE_MODELS: List[str] = [
    # OpenAI (5 models)
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    # Google (4 models)
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    # Anthropic (5 models)
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    # Qwen (1 model)
    "qwen3.8-max",
    # ZAI / Zhipu (2 models)
    "glm-5.3",
    "glm-5.3-flash",
    # Moonshot AI (1 model)
    "kimi-k3",
]

NUM_CLASSES: int = len(CANDIDATE_MODELS)  # 18
MODEL_TO_IDX: Dict[str, int] = {model: i for i, model in enumerate(CANDIDATE_MODELS)}
IDX_TO_MODEL: Dict[int, str] = {i: model for i, model in enumerate(CANDIDATE_MODELS)}

# ---------------------------------------------------------------------------
# AI Lab / Family Taxonomy (Hierarchical Gating)
# ---------------------------------------------------------------------------
MODEL_FAMILY_MAP: Dict[str, str] = {
    # OpenAI
    "gpt-5.5": "OpenAI",
    "gpt-5.6-luna": "OpenAI",
    "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI",
    "gpt-6-astra": "OpenAI",
    # Google
    "gemini-3.1-pro-preview": "Google",
    "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google",
    "gemini-3.8-flash": "Google",
    # Anthropic
    "claude-fable-5": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic",
    "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    # Qwen
    "qwen3.8-max": "Qwen",
    # ZAI
    "glm-5.3": "ZAI",
    "glm-5.3-flash": "ZAI",
    # Moonshot
    "kimi-k3": "Moonshot",
}

FAMILY_NAMES: List[str] = ["OpenAI", "Anthropic", "Google", "ZAI", "Qwen", "Moonshot"]
FAMILY_TO_IDX: Dict[str, int] = {fam: i for i, fam in enumerate(FAMILY_NAMES)}
IDX_TO_FAMILY: Dict[int, str] = {i: fam for i, fam in enumerate(FAMILY_NAMES)}
NUM_FAMILIES: int = len(FAMILY_NAMES)

# Mapping from family to model indices
FAMILY_TO_MODEL_INDICES: Dict[str, List[int]] = {}
for fam in FAMILY_NAMES:
    FAMILY_TO_MODEL_INDICES[fam] = [
        MODEL_TO_IDX[m] for m, f in MODEL_FAMILY_MAP.items() if f == fam
    ]

# ---------------------------------------------------------------------------
# Paths and Environment Defaults
# ---------------------------------------------------------------------------
DEFAULT_KAGGLE_INPUT_PATHS: List[Path] = [
    Path("/kaggle/input/name-that-llm"),
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("./data"),
    Path("../data"),
]

SUBMISSION_FILENAME: str = "submission.parquet"
RANDOM_SEED: int = 42
N_FOLDS: int = 5
TARGET_COLUMN: str = "model"
PROMPT_GROUP_COLUMN: str = "prompt_id"
TEXT_COLUMN: str = "text"
ID_COLUMN: str = "id"
