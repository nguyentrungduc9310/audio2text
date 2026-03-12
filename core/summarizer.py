import logging
from typing import Optional

from core.llm_provider import LLMProvider, LLMResponse
from core.text_splitter import split_text, chars_for_tokens

logger = logging.getLogger(__name__)

PROMPT_OVERHEAD_TOKENS = 400
OUTPUT_RESERVE_TOKENS = 2048
CHUNK_RATIO = 0.7

LANG_NAMES = {
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "th": "Thai",
    "en": "English",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "id": "Indonesian",
}

SYSTEM_PROMPT = """You are a professional meeting/audio transcript summarizer.

Rules:
- Summarize in the SAME language as the input text
- Use structured bullet points
- Capture key topics, decisions, action items, and important details
- Be concise but comprehensive
- Do NOT translate — keep the original language"""

MERGE_SYSTEM_PROMPT = """You are a professional summarizer.

Rules:
- Merge the following partial summaries into one cohesive summary
- Use structured bullet points
- Remove duplicates and organize by topic
- Keep the SAME language as the input
- Do NOT translate"""


def _lang_hint(source_lang: Optional[str]) -> str:
    """Return a language instruction string to append to system prompts."""
    if not source_lang:
        return ""
    language_name = LANG_NAMES.get(source_lang, source_lang)
    return f"\n- IMPORTANT: The input text is in {language_name}. Your summary MUST be written in {language_name}."


def _compute_chunk_budget(provider: LLMProvider, lang: str) -> int:
    """Compute max chars per chunk based on provider's context window."""
    ctx = provider.context_window()
    available_tokens = ctx - PROMPT_OVERHEAD_TOKENS - OUTPUT_RESERVE_TOKENS
    token_budget = int(available_tokens * CHUNK_RATIO)
    budget = chars_for_tokens(max(token_budget, 1000), lang or "en")
    logger.info("Summarizer chunk budget: %d chars (context=%d, lang=%s)", budget, ctx, lang)
    return budget


class Summarizer:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def summarize(self, text: str, model: Optional[str] = None, on_token=None, source_lang: Optional[str] = None) -> LLMResponse:
        lang = source_lang or "en"
        budget = _compute_chunk_budget(self.provider, lang)

        if len(text) <= budget:
            return self._summarize_single(text, model, on_token, source_lang)
        return self._summarize_chunked(text, model, on_token, source_lang, budget)

    def _summarize_single(self, text: str, model: Optional[str] = None, on_token=None, source_lang: Optional[str] = None) -> LLMResponse:
        prompt = f"Summarize the following transcript:\n\n{text}"

        logger.info("Summarizing text (single pass, %d chars)", len(text))

        system_prompt = SYSTEM_PROMPT + _lang_hint(source_lang)

        return self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            on_token=on_token,
        )

    def _summarize_chunked(self, text: str, model: Optional[str] = None, on_token=None, source_lang: Optional[str] = None, budget: int = 4000) -> LLMResponse:
        chunks = split_text(text, budget, boundary="sentence", overlap_sentences=2)

        logger.info("Summarizing text (chunked, %d chunks, budget=%d chars)", len(chunks), budget)

        hint = _lang_hint(source_lang)
        chunk_system_prompt = SYSTEM_PROMPT + hint

        partial_summaries = []
        for i, chunk in enumerate(chunks, 1):
            logger.info("Summarizing chunk %d/%d (%d chars)", i, len(chunks), len(chunk))

            prompt = f"Summarize part {i}/{len(chunks)} of the transcript:\n\n{chunk}"

            response = self.provider.generate(
                prompt=prompt,
                system_prompt=chunk_system_prompt,
                model=model,
                on_token=on_token,
            )
            partial_summaries.append(response.text)

        # Safe merge: re-chunk if merged text exceeds budget
        merged_text = "\n\n---\n\n".join(
            f"Part {i}: {s}" for i, s in enumerate(partial_summaries, 1)
        )

        merge_system_prompt = MERGE_SYSTEM_PROMPT + hint

        if len(merged_text) <= budget:
            prompt = f"Merge these partial summaries into one cohesive summary:\n\n{merged_text}"
            logger.info("Merging %d partial summaries (single pass, %d chars)", len(partial_summaries), len(merged_text))
            return self.provider.generate(
                prompt=prompt,
                system_prompt=merge_system_prompt,
                model=model,
                on_token=on_token,
            )

        # Recursive merge: re-chunk partial summaries and summarize again
        logger.info("Merged text too large (%d chars > %d budget), re-chunking for recursive merge", len(merged_text), budget)
        merge_chunks = split_text(merged_text, budget, boundary="sentence", overlap_sentences=0)
        re_summaries = []
        for i, mc in enumerate(merge_chunks, 1):
            logger.info("Re-summarizing merge chunk %d/%d (%d chars)", i, len(merge_chunks), len(mc))
            response = self.provider.generate(
                prompt=f"Consolidate these partial summaries into a single summary:\n\n{mc}",
                system_prompt=merge_system_prompt,
                model=model,
            )
            re_summaries.append(response.text)

        final_merged = "\n\n---\n\n".join(re_summaries)
        prompt = f"Merge these summaries into one final cohesive summary:\n\n{final_merged}"
        logger.info("Final merge of %d re-summaries (%d chars)", len(re_summaries), len(final_merged))
        return self.provider.generate(
            prompt=prompt,
            system_prompt=merge_system_prompt,
            model=model,
            on_token=on_token,
        )
