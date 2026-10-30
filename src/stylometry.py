"""
Stylometric and lexical feature extraction engine.
Extracts 120+ specialized stylistic fingerprints, punctuation profiles,
lexical diversity metrics, and discourse markers for LLM authorship attribution.
"""

from collections import Counter
import math
import re
from typing import Dict, List, Union
import numpy as np
import pandas as pd


# Precompiled regex patterns for speed
RE_SENTENCE = re.compile(r"[.!?]+(?:\s+|$)")
RE_WORD = re.compile(r"\b[a-zA-Z0-9_'-]+\b")
RE_OXFORD_COMMA = re.compile(r",\s+(?:and|or|nor|but)\b", re.IGNORECASE)
RE_NUMBERED_LIST = re.compile(r"(?:^|\n)\s*\d+[.)]\s+")
RE_BULLET_LIST = re.compile(r"(?:^|\n)\s*[-*•+]\s+")
RE_HEADER = re.compile(r"(?:^|\n)\s*#{1,6}\s+")
RE_BOLD = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
RE_ITALIC = re.compile(r"(?<!\*)\*[^*]+\*(?!\*)|(?<!_)_[^_]+_(?!_)")
RE_INLINE_CODE = re.compile(r"`[^`]+`")
RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
RE_CONTRACTION = re.compile(
    r"\b(?:don't|can't|it's|you're|we've|i'm|doesn't|won't|isn't|aren't|haven't|hadn't|couldn't|shouldn't|wouldn't|that's|there's|what's|who's|let's)\b",
    re.IGNORECASE,
)
RE_EXPANDED_FORM = re.compile(
    r"\b(?:do not|cannot|can not|it is|you are|we have|i am|does not|will not|is not|are not|have not|had not|could not|should not|would not|that is|there is|what is|who is|let us)\b",
    re.IGNORECASE,
)

# Characteristic discourse markers observed across LLM labs
DISCOURSE_MARKERS = [
    "furthermore",
    "moreover",
    "in conclusion",
    "it is worth noting",
    "it is important to note",
    "crucially",
    "delve",
    "tapestry",
    "indeed",
    "however",
    "consequently",
    "specifically",
    "additionally",
    "notably",
    "essentially",
    "ultimately",
    "let's explore",
    "to understand this",
    "importantly",
    "in summary",
    "in essence",
    "on the other hand",
    "at its core",
    "step-by-step",
    "first and foremost",
    "similarly",
    "conversely",
    "nevertheless",
    "overall",
    "in contrast",
]

PRONOUNS_1ST = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
PRONOUNS_2ND = {"you", "your", "yours", "yourself", "yourselves"}
PRONOUNS_3RD = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}
BE_VERBS = {"is", "am", "are", "was", "were", "be", "being", "been"}


class StylometryExtractor:
    """
    High-performance feature extractor computing fine-grained stylometric,
    typographic, structural, and lexical fingerprints from text passages.
    """

    def __init__(self):
        self.feature_names_: List[str] = []

    def extract_single(self, text: str) -> Dict[str, float]:
        """Extracts all stylometric features for a single passage string."""
        if not isinstance(text, str) or not text.strip():
            text = " "

        feats: Dict[str, float] = {}
        text_len = len(text)
        text_lower = text.lower()

        # -------------------------------------------------------------------
        # 1. Length and Volume Metrics
        # -------------------------------------------------------------------
        words = RE_WORD.findall(text)
        word_count = len(words)
        words_lower = [w.lower() for w in words]

        sentences = [s.strip() for s in RE_SENTENCE.split(text) if s.strip()]
        sentence_count = max(1, len(sentences))

        lines = text.split("\n")
        line_count = len(lines)
        blank_line_count = sum(1 for line in lines if not line.strip())
        non_blank_lines = [l for l in lines if l.strip()]
        paragraph_count = max(1, len(non_blank_lines))

        feats["char_count"] = float(text_len)
        feats["word_count"] = float(word_count)
        feats["sentence_count"] = float(sentence_count)
        feats["line_count"] = float(line_count)
        feats["paragraph_count"] = float(paragraph_count)
        feats["blank_line_ratio"] = blank_line_count / max(1, line_count)

        # Average lengths & variability
        safe_words = max(1, word_count)
        feats["avg_word_length"] = float(np.mean([len(w) for w in words])) if words else 0.0
        feats["std_word_length"] = float(np.std([len(w) for w in words])) if words else 0.0

        sent_lens = [len(s.split()) for s in sentences]
        feats["avg_sentence_length_words"] = float(np.mean(sent_lens)) if sent_lens else 0.0
        feats["std_sentence_length_words"] = float(np.std(sent_lens)) if sent_lens else 0.0
        feats["max_sentence_length_words"] = float(np.max(sent_lens)) if sent_lens else 0.0
        feats["min_sentence_length_words"] = float(np.min(sent_lens)) if sent_lens else 0.0

        para_lens = [len(p.split()) for p in non_blank_lines]
        feats["avg_paragraph_length_words"] = float(np.mean(para_lens)) if para_lens else 0.0
        feats["std_paragraph_length_words"] = float(np.std(para_lens)) if para_lens else 0.0

        # -------------------------------------------------------------------
        # 2. Punctuation & Typographic Signatures
        # -------------------------------------------------------------------
        # Exact frequencies per word
        feats["punct_comma"] = text.count(",") / safe_words
        feats["punct_period"] = text.count(".") / safe_words
        feats["punct_semicolon"] = text.count(";") / safe_words
        feats["punct_colon"] = text.count(":") / safe_words
        feats["punct_exclamation"] = text.count("!") / safe_words
        feats["punct_question"] = text.count("?") / safe_words

        # Dashes (Em-dash, En-dash, Hyphen)
        feats["punct_em_dash"] = text.count("—") / safe_words
        feats["punct_en_dash"] = text.count("–") / safe_words
        feats["punct_double_hyphen"] = text.count("--") / safe_words
        feats["punct_single_hyphen"] = text.count("-") / safe_words
        feats["punct_all_dashes"] = (
            text.count("—") + text.count("–") + text.count("--") + text.count("-")
        ) / safe_words

        # Quotes and backticks
        feats["punct_straight_double_quote"] = text.count('"') / safe_words
        feats["punct_curly_double_quotes"] = (text.count("“") + text.count("”")) / safe_words
        feats["punct_straight_single_quote"] = text.count("'") / safe_words
        feats["punct_curly_single_quotes"] = (text.count("‘") + text.count("’")) / safe_words
        feats["punct_backtick"] = text.count("`") / safe_words

        # Brackets & Delimiters
        feats["punct_paren_open"] = text.count("(") / safe_words
        feats["punct_paren_close"] = text.count(")") / safe_words
        feats["punct_bracket_open"] = text.count("[") / safe_words
        feats["punct_bracket_close"] = text.count("]") / safe_words
        feats["punct_brace_open"] = text.count("{") / safe_words
        feats["punct_brace_close"] = text.count("}") / safe_words

        # Ellipses & Oxford Comma
        feats["punct_dots_ellipsis"] = text.count("...") / safe_words
        feats["punct_unicode_ellipsis"] = text.count("…") / safe_words
        feats["punct_oxford_comma"] = len(RE_OXFORD_COMMA.findall(text)) / safe_words

        # Ratios
        comma_count = max(1, text.count(","))
        feats["ratio_semicolon_to_comma"] = text.count(";") / comma_count
        feats["ratio_colon_to_sentence"] = text.count(":") / sentence_count

        # -------------------------------------------------------------------
        # 3. Markdown and Layout Signatures
        # -------------------------------------------------------------------
        feats["md_bullet_items"] = len(RE_BULLET_LIST.findall(text)) / sentence_count
        feats["md_numbered_items"] = len(RE_NUMBERED_LIST.findall(text)) / sentence_count
        feats["md_headers"] = len(RE_HEADER.findall(text)) / sentence_count
        feats["md_bold_matches"] = len(RE_BOLD.findall(text)) / sentence_count
        feats["md_italic_matches"] = len(RE_ITALIC.findall(text)) / sentence_count
        feats["md_inline_code"] = len(RE_INLINE_CODE.findall(text)) / sentence_count
        feats["md_code_blocks"] = len(RE_CODE_BLOCK.findall(text)) / sentence_count

        # -------------------------------------------------------------------
        # 4. Capitalization & Character Distributions
        # -------------------------------------------------------------------
        safe_chars = max(1, text_len)
        feats["cap_uppercase_ratio"] = sum(1 for c in text if c.isupper()) / safe_chars
        feats["cap_digit_ratio"] = sum(1 for c in text if c.isdigit()) / safe_chars
        feats["cap_whitespace_ratio"] = sum(1 for c in text if c.isspace()) / safe_chars
        feats["cap_all_caps_words"] = (
            sum(1 for w in words if w.isupper() and len(w) > 1) / safe_words
        )
        feats["cap_title_case_words"] = (
            sum(1 for w in words if w.istitle() and len(w) > 1) / safe_words
        )

        # -------------------------------------------------------------------
        # 5. Lexical Diversity & Stylometric Invariants
        # -------------------------------------------------------------------
        if words_lower:
            vocab = set(words_lower)
            vocab_size = len(vocab)
            freqs = Counter(words_lower)

            # Type-Token Ratio (TTR)
            feats["lex_ttr"] = vocab_size / safe_words

            # Root TTR (Guiraud's Index)
            feats["lex_guiraud"] = vocab_size / math.sqrt(safe_words)

            # Hapax Legomena (words occurring exactly once)
            hapax_count = sum(1 for count in freqs.values() if count == 1)
            feats["lex_hapax_ratio"] = hapax_count / safe_words

            # Dis Legomena (words occurring exactly twice)
            dis_count = sum(1 for count in freqs.values() if count == 2)
            feats["lex_dis_ratio"] = dis_count / safe_words

            # Yule's Characteristic K
            # K = 10^4 * (sum(m^2 * V_m) - N) / N^2
            freq_spectrum = Counter(freqs.values())
            sum_m2_vm = sum((m**2) * vm for m, vm in freq_spectrum.items())
            if safe_words > 1:
                yule_k = 10000.0 * (sum_m2_vm - safe_words) / (safe_words**2)
            else:
                yule_k = 0.0
            feats["lex_yule_k"] = float(yule_k)

            # Simpson's Diversity Index
            if safe_words > 1:
                simpson = sum(n * (n - 1) for n in freqs.values()) / (safe_words * (safe_words - 1))
            else:
                simpson = 1.0
            feats["lex_simpson"] = float(simpson)
        else:
            feats["lex_ttr"] = 0.0
            feats["lex_guiraud"] = 0.0
            feats["lex_hapax_ratio"] = 0.0
            feats["lex_dis_ratio"] = 0.0
            feats["lex_yule_k"] = 0.0
            feats["lex_simpson"] = 0.0

        # -------------------------------------------------------------------
        # 6. Discourse Markers & Tone
        # -------------------------------------------------------------------
        for marker in DISCOURSE_MARKERS:
            key = f"discourse_{marker.replace(' ', '_').replace('-', '_')}"
            count = text_lower.count(marker)
            feats[key] = count / safe_words

        # Pronoun distributions
        feats["pronoun_1st"] = sum(1 for w in words_lower if w in PRONOUNS_1ST) / safe_words
        feats["pronoun_2nd"] = sum(1 for w in words_lower if w in PRONOUNS_2ND) / safe_words
        feats["pronoun_3rd"] = sum(1 for w in words_lower if w in PRONOUNS_3RD) / safe_words

        # Passive/auxiliary verb density
        feats["aux_be_verbs"] = sum(1 for w in words_lower if w in BE_VERBS) / safe_words

        # Contractions vs expanded forms
        contractions_count = len(RE_CONTRACTION.findall(text))
        expanded_count = len(RE_EXPANDED_FORM.findall(text))
        feats["count_contractions"] = contractions_count / safe_words
        feats["count_expanded_forms"] = expanded_count / safe_words
        feats["ratio_contraction_to_expanded"] = contractions_count / max(1, contractions_count + expanded_count)

        return feats

    def fit(self, texts: List[str]):
        """Fits extractor and stores ordered feature names."""
        sample_feats = self.extract_single(texts[0] if texts else "Sample text")
        self.feature_names_ = sorted(sample_feats.keys())
        return self

    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Transforms a collection of text passages into a dense feature matrix."""
        if not self.feature_names_:
            self.fit([str(texts[0]) if len(texts) > 0 else "Sample text"])

        rows = []
        for text in texts:
            fdict = self.extract_single(str(text))
            row = [fdict.get(fn, 0.0) for fn in self.feature_names_]
            rows.append(row)

        X = np.array(rows, dtype=np.float32)
        # Clean any potential NaN or Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        self.fit(list(texts))
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.feature_names_
