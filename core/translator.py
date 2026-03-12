import logging
from typing import Optional, List, Dict

from core.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

CHUNK_SIZE = 3000

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

        if len(text) <= CHUNK_SIZE:
            return self._translate_single(text, system_prompt, model, on_token)
        return self._translate_chunked(text, system_prompt, model, on_token)

    def _translate_single(self, text: str, system_prompt: str, model: Optional[str] = None, on_token=None) -> LLMResponse:
        logger.info("Translating text (single pass, %d chars)", len(text))

        return self.provider.generate(
            prompt=f"Translate:\n\n{text}",
            system_prompt=system_prompt,
            model=model,
            on_token=on_token,
        )

    def _translate_chunked(self, text: str, system_prompt: str, model: Optional[str] = None, on_token=None) -> LLMResponse:
        paragraphs = text.split("\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 > CHUNK_SIZE and current:
                chunks.append(current)
                current = para
            else:
                current = current + "\n" + para if current else para

        if current:
            chunks.append(current)

        logger.info("Translating text (chunked, %d chunks)", len(chunks))

        translated_parts = []
        for i, chunk in enumerate(chunks, 1):
            logger.info("Translating chunk %d/%d", i, len(chunks))

            response = self.provider.generate(
                prompt=f"Translate part {i}/{len(chunks)}:\n\n{chunk}",
                system_prompt=system_prompt,
                model=model,
                on_token=on_token,
            )
            translated_parts.append(response.text)

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

        # Batch segments into chunks for efficiency
        batches = []
        current_batch = []
        current_len = 0

        for seg in segments:
            seg_text = seg.get("text", "")
            if current_len + len(seg_text) > CHUNK_SIZE and current_batch:
                batches.append(current_batch)
                current_batch = [seg]
                current_len = len(seg_text)
            else:
                current_batch.append(seg)
                current_len += len(seg_text)

        if current_batch:
            batches.append(current_batch)

        logger.info("Translating %d segments in %d batches", len(segments), len(batches))

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
