import logging
from typing import Optional

from core.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4000

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


class Summarizer:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def summarize(self, text: str, model: Optional[str] = None, on_token=None, source_lang: Optional[str] = None) -> LLMResponse:
        if len(text) <= CHUNK_SIZE:
            return self._summarize_single(text, model, on_token, source_lang)
        return self._summarize_chunked(text, model, on_token, source_lang)

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

    def _summarize_chunked(self, text: str, model: Optional[str] = None, on_token=None, source_lang: Optional[str] = None) -> LLMResponse:
        chunks = []
        for i in range(0, len(text), CHUNK_SIZE):
            chunks.append(text[i:i + CHUNK_SIZE])

        logger.info("Summarizing text (chunked, %d chunks)", len(chunks))

        hint = _lang_hint(source_lang)
        chunk_system_prompt = SYSTEM_PROMPT + hint

        partial_summaries = []
        for i, chunk in enumerate(chunks, 1):
            logger.info("Summarizing chunk %d/%d", i, len(chunks))

            prompt = f"Summarize part {i}/{len(chunks)} of the transcript:\n\n{chunk}"

            response = self.provider.generate(
                prompt=prompt,
                system_prompt=chunk_system_prompt,
                model=model,
                on_token=on_token,
            )
            partial_summaries.append(response.text)

        merged_text = "\n\n---\n\n".join(
            f"Part {i}: {s}" for i, s in enumerate(partial_summaries, 1)
        )

        prompt = f"Merge these partial summaries into one cohesive summary:\n\n{merged_text}"

        logger.info("Merging %d partial summaries", len(partial_summaries))

        merge_system_prompt = MERGE_SYSTEM_PROMPT + hint

        return self.provider.generate(
            prompt=prompt,
            system_prompt=merge_system_prompt,
            model=model,
            on_token=on_token,
        )
