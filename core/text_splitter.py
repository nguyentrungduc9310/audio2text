"""Shared text splitting utilities for chunked LLM processing."""

import re
from typing import List

# Approximate chars-per-token ratios by language group
_CHARS_PER_TOKEN = {
    "en": 4.0, "fr": 4.0, "de": 4.0, "es": 4.0, "it": 4.0,
    "nl": 4.0, "pl": 4.0, "pt": 4.0, "tr": 4.0, "id": 4.0,
    "vi": 3.0, "th": 3.0,
    "zh": 1.5, "ja": 1.5, "ko": 1.5,
    "ar": 3.0, "hi": 3.0, "ru": 3.0,
}
_DEFAULT_CHARS_PER_TOKEN = 3.5

# Sentence-ending pattern: standard punctuation + CJK sentence enders
_SENTENCE_END = re.compile(r'(?<=[.!?。！？])\s+')


def estimate_tokens(text: str, lang: str = "en") -> int:
    """Estimate token count for text based on language-specific char/token ratio."""
    ratio = _CHARS_PER_TOKEN.get(lang, _DEFAULT_CHARS_PER_TOKEN)
    return max(1, int(len(text) / ratio))


def chars_for_tokens(token_budget: int, lang: str = "en") -> int:
    """Convert a token budget to approximate character count."""
    ratio = _CHARS_PER_TOKEN.get(lang, _DEFAULT_CHARS_PER_TOKEN)
    return int(token_budget * ratio)


def split_text(
    text: str,
    max_chars: int,
    boundary: str = "sentence",
    overlap_sentences: int = 0,
) -> List[str]:
    """Split text into chunks respecting sentence or paragraph boundaries.

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk.
        boundary: "sentence" or "paragraph".
        overlap_sentences: Number of trailing sentences from previous chunk
            to prepend to the next chunk (only for sentence boundary).

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chars:
        return [text]

    if boundary == "paragraph":
        units = text.split("\n")
        joiner = "\n"
    else:
        units = _SENTENCE_END.split(text)
        joiner = " "

    # Remove empty units
    units = [u for u in units if u.strip()]

    if not units:
        return [text]

    chunks = []
    current_units: List[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit) + len(joiner)

        if current_len + unit_len > max_chars and current_units:
            chunks.append(joiner.join(current_units))

            # Overlap: carry last N sentences to next chunk
            if overlap_sentences > 0 and boundary == "sentence":
                current_units = current_units[-overlap_sentences:]
                current_len = sum(len(u) + len(joiner) for u in current_units)
            else:
                current_units = []
                current_len = 0

        current_units.append(unit)
        current_len += unit_len

    if current_units:
        chunk = joiner.join(current_units)
        # Avoid duplicating the last chunk if overlap made it identical
        if not chunks or chunk != chunks[-1]:
            chunks.append(chunk)

    return chunks
