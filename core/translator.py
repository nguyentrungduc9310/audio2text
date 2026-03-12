import logging
from typing import Optional, List, Dict

from core.llm_provider import LLMProvider, LLMResponse
from core.text_splitter import split_text, chars_for_tokens

logger = logging.getLogger(__name__)

PROMPT_OVERHEAD_TOKENS = 400
OUTPUT_RESERVE_TOKENS = 2048
TRANSLATE_RATIO = 0.45  # Lower than summarizer since translation output ≈ input size
CONTEXT_HINT_CHARS = 200

MEETING_CONTEXTS = {
    "general": "general meetings and conversations",
    "technical": "software engineering, DevOps, cloud computing, and AI technical meetings",
    "sales": "sales, business development, products, customers, and revenue discussions",
    "medical": "healthcare, medicine, and pharmaceutical discussions",
    "legal": "legal proceedings, contracts, and regulatory discussions",
}

TARGET_LANGUAGES = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "th": "Thai",
}


def _build_system_prompt(source_lang: str, target_lang: str, meeting_context: str, glossary: Optional[Dict[str, str]] = None) -> str:
    target_name = TARGET_LANGUAGES.get(target_lang, target_lang)
    context_desc = MEETING_CONTEXTS.get(meeting_context, MEETING_CONTEXTS["general"])

    prompt = f"""You are a professional translator specializing in {context_desc}.

Translate the text to {target_name}.

Rules:
- Produce ONLY the translated text, no explanations or notes
- Preserve the structure and formatting of the original text
- Preserve technical terms that are commonly used in their original form in {target_name} context"""

    if meeting_context == "technical":
        prompt += """
- Keep terms like "pipeline", "container", "deployment", "API", "microservice", "Kubernetes", "Docker" in their original form when commonly used as-is"""
    elif meeting_context == "sales":
        prompt += """
- Translate business terms naturally but keep product names and brand names as-is"""
    elif meeting_context == "medical":
        prompt += """
- Use standard medical terminology in the target language
- Keep drug names and medical abbreviations as-is"""
    elif meeting_context == "legal":
        prompt += """
- Use formal legal terminology in the target language
- Keep legal Latin terms as-is"""

    if glossary:
        prompt += "\n\nGlossary (use these translations):\n"
        for src, tgt in glossary.items():
            prompt += f'- "{src}" → "{tgt}"\n'

    return prompt


def _compute_chunk_budget(provider: LLMProvider, lang: str) -> int:
    """Compute max chars per chunk based on provider's context window."""
    ctx = provider.context_window()
    available_tokens = ctx - PROMPT_OVERHEAD_TOKENS - OUTPUT_RESERVE_TOKENS
    token_budget = int(available_tokens * TRANSLATE_RATIO)
    budget = chars_for_tokens(max(token_budget, 1000), lang or "en")
    logger.info("Translator chunk budget: %d chars (context=%d, lang=%s)", budget, ctx, lang)
    return budget


class Translator:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "vi",
        meeting_context: str = "general",
        glossary: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        on_token=None,
    ) -> LLMResponse:
        system_prompt = _build_system_prompt(source_lang, target_lang, meeting_context, glossary)
        budget = _compute_chunk_budget(self.provider, source_lang)

        if len(text) <= budget:
            return self._translate_single(text, system_prompt, model, on_token)
        return self._translate_chunked(text, system_prompt, model, on_token, budget)

    def _translate_single(self, text: str, system_prompt: str, model: Optional[str] = None, on_token=None) -> LLMResponse:
        logger.info("Translating text (single pass, %d chars)", len(text))

        return self.provider.generate(
            prompt=f"Translate:\n\n{text}",
            system_prompt=system_prompt,
            model=model,
            on_token=on_token,
        )

    def _translate_chunked(self, text: str, system_prompt: str, model: Optional[str] = None, on_token=None, budget: int = 3000) -> LLMResponse:
        # Split by paragraph first, then fall back to sentence for oversized paragraphs
        chunks = split_text(text, budget, boundary="paragraph")
        # Re-split any oversized paragraph chunks by sentence
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > budget:
                final_chunks.extend(split_text(chunk, budget, boundary="sentence"))
            else:
                final_chunks.append(chunk)
        chunks = final_chunks

        logger.info("Translating text (chunked, %d chunks, budget=%d chars)", len(chunks), budget)

        translated_parts = []
        prev_tail = ""

        for i, chunk in enumerate(chunks, 1):
            logger.info("Translating chunk %d/%d (%d chars)", i, len(chunks), len(chunk))

            # Add context hint from previous translation for terminology consistency
            context_hint = ""
            if prev_tail:
                context_hint = f"[Context — end of previous translated section: ...{prev_tail}]\n\n"

            prompt = f"{context_hint}Translate part {i}/{len(chunks)}:\n\n{chunk}"

            response = self.provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                on_token=on_token,
            )
            translated_parts.append(response.text)
            prev_tail = response.text[-CONTEXT_HINT_CHARS:] if len(response.text) > CONTEXT_HINT_CHARS else response.text

        final_text = "\n".join(translated_parts)

        return LLMResponse(
            text=final_text,
            model=response.model,
            provider=response.provider,
            tokens_used=None,
        )

    def translate_segments(
        self,
        segments: List[Dict],
        source_lang: str = "en",
        target_lang: str = "vi",
        meeting_context: str = "general",
        glossary: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> List[Dict]:
        if not segments:
            return []

        system_prompt = _build_system_prompt(source_lang, target_lang, meeting_context, glossary)
        budget = _compute_chunk_budget(self.provider, source_lang)

        # Batch segments into chunks for efficiency
        batches = []
        current_batch = []
        current_len = 0

        for seg in segments:
            seg_text = seg.get("text", "")
            if current_len + len(seg_text) > budget and current_batch:
                batches.append(current_batch)
                current_batch = [seg]
                current_len = len(seg_text)
            else:
                current_batch.append(seg)
                current_len += len(seg_text)

        if current_batch:
            batches.append(current_batch)

        logger.info("Translating %d segments in %d batches (budget=%d chars)", len(segments), len(batches), budget)

        translated_segments = []

        for i, batch in enumerate(batches, 1):
            logger.info("Translating segment batch %d/%d (%d segments)", i, len(batches), len(batch))

            numbered_lines = []
            for j, seg in enumerate(batch):
                numbered_lines.append(f"[{j}] {seg.get('text', '')}")

            prompt = "Translate each numbered line. Keep the [N] prefix for each line:\n\n" + "\n".join(numbered_lines)

            response = self.provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
            )

            # Parse numbered translations
            translations = {}
            for line in response.text.strip().split("\n"):
                line = line.strip()
                if line.startswith("[") and "]" in line:
                    bracket_end = line.index("]")
                    try:
                        idx = int(line[1:bracket_end])
                        text = line[bracket_end + 1:].strip()
                        translations[idx] = text
                    except ValueError:
                        continue

            for j, seg in enumerate(batch):
                translated_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": translations.get(j, seg.get("text", "")),
                })

        return translated_segments
