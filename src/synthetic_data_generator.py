"""
Synthetic data generator utility for local pipeline verification and testing.
Creates realistic mock datasets adhering to the exact schema of 'Name That LLM'.
"""

import random
from typing import Dict, List, Tuple
import pandas as pd

from .config import CANDIDATE_MODELS, MODEL_FAMILY_MAP


# Sample domain prompts
PROMPTS = [
    "Explain the implications of the second law of thermodynamics on entropy.",
    "Discuss the philosophical problem of other minds and conscious experience.",
    "Describe the algorithm for finding strongly connected components in directed graphs.",
    "Analyze the role of mitochondria in cellular energy production and ATP synthesis.",
    "Write a short narrative capturing an astronaut's perspective during orbital sunrise.",
    "Compare utilitarianism and deontological ethics in autonomous vehicle decision-making.",
    "Explain the mathematical intuition behind Fourier transforms in signal processing.",
    "Detail the mechanism of CRISPR-Cas9 genome editing and guide RNA design.",
]

# Style templates mimicking each family's stylistic habits
FAMILY_STYLE_SNIPPETS = {
    "OpenAI": [
        "It is worth noting that {subject}. When examining this system, several key factors emerge: First, the underlying mechanism operates predictably. Furthermore, the interplay between variables ensures balance—an essential consideration for practical implementations. In conclusion, this perspective offers a robust framework.",
        "To understand this concept thoroughly, we should consider both theoretical and applied dimensions. Crucially, {subject}. On the other hand, empirical constraints cannot be overlooked. Ultimately, careful analysis reveals a clear and harmonious structure.",
    ],
    "Anthropic": [
        "When considering {subject}, one is struck by the nuanced interplay of foundational principles. There is—perhaps counterintuitively—a profound elegance in how these constraints manifest; not merely as boundaries, but as generative structures. It is helpful to reflect on whether this distinction holds under closer inspection.",
        "The question invites a thoughtful examination of {subject}. In philosophical terms, we might delineate two complementary aspects: the formal architecture and its interpretive ramifications (which often resist simple categorization). Consequently, any comprehensive account must remain open to inherent ambiguities.",
    ],
    "Google": [
        "**Core Overview:**\n{subject}.\n\nKey takeaways include:\n* **Mechanism:** Direct execution with optimized feedback loops.\n* **Efficiency:** High-throughput processing across components.\n* **Outcome:** Predictable and measurable results.",
        "Here is a breakdown of {subject}:\n1. **Foundational Principle:** Grounded in fundamental physical laws.\n2. **Practical Application:** Streamlines analytical pipelines.\n3. **Summary:** Delivers consistent accuracy with minimal latency.",
    ],
    "ZAI": [
        "Specifically, regarding {subject}, the formal analysis demonstrates distinct characteristics. Furthermore, from an analytical standpoint, the transformation satisfies standard boundary conditions. In terms of systemic properties, the outcome remains invariant across typical scenarios.",
        "To systematically address {subject}: Firstly, the theoretical basis is established via direct derivation. Additionally, the empirical validation indicates consistent alignment with expected parameters. In summary, the findings provide a rigorous baseline.",
    ],
    "Qwen": [
        "Analyzing {subject} requires a structured methodology:\n\nStep 1: Define the operational boundaries and invariant state variables.\nStep 2: Apply the governing transformation equations directly.\nStep 3: Evaluate computational complexity, which scales linearly with input dimensions.\nTherefore, the formulation is both computationally efficient and mathematically sound.",
        "The problem of {subject} can be formally resolved as follows. By decomposing the system into discrete sub-problems, each component is addressed independently. Consequently, the synthesized solution maintains formal correctness throughout.",
    ],
    "Moonshot": [
        "Exploring {subject} reveals fascinating insights across multiple layers of context. At its core, the dynamic relationships between components illuminate how complex behaviors emerge from straightforward rules. In essence, synthesizing these viewpoints provides a well-rounded and actionable understanding.",
        "When we look at {subject} from a holistic angle, the broader implications become remarkably clear. Rather than isolated events, each phase represents a continuous progression. Ultimately, this comprehensive perspective bridges theory and practice.",
    ],
}


def generate_synthetic_dataset(
    n_prompts: int = 10,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a synthetic (train_df, test_df) pair conforming to Kaggle schema:
    - train_df columns: ['id', 'prompt_id', 'text', 'model']
    - test_df columns: ['id', 'prompt_id', 'text']
    """
    random.seed(random_seed)

    train_rows = []
    test_rows = []

    # Generate training data: all 18 models answer each prompt
    for p_idx in range(n_prompts):
        prompt_id = f"prompt_{p_idx:03d}"
        prompt_topic = PROMPTS[p_idx % len(PROMPTS)]

        for m_idx, model_name in enumerate(CANDIDATE_MODELS):
            family = MODEL_FAMILY_MAP[model_name]
            templates = FAMILY_STYLE_SNIPPETS[family]
            template = random.choice(templates)
            passage = template.format(subject=prompt_topic.rstrip("."))

            row_id = f"train_{p_idx:03d}_{m_idx:02d}"
            train_rows.append({
                "id": row_id,
                "prompt_id": prompt_id,
                "text": passage,
                "model": model_name,
            })

    # Generate synthetic test data: 3 unseen prompts
    for p_idx in range(n_prompts, n_prompts + 3):
        prompt_id = f"prompt_{p_idx:03d}"
        prompt_topic = PROMPTS[p_idx % len(PROMPTS)]

        # Random subset of models answering test prompts (realistic test variation)
        test_models = random.sample(CANDIDATE_MODELS, k=random.randint(12, 18))
        for m_idx, model_name in enumerate(test_models):
            family = MODEL_FAMILY_MAP[model_name]
            template = random.choice(FAMILY_STYLE_SNIPPETS[family])
            passage = template.format(subject=prompt_topic.rstrip("."))

            row_id = f"test_{p_idx:03d}_{m_idx:02d}"
            test_rows.append({
                "id": row_id,
                "prompt_id": prompt_id,
                "text": passage,
            })

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, test_df
