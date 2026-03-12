import logging
from typing import Optional, List

from core.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "context_window": 128000,
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-5-20250929",
        "context_window": 200000,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "context_window": 1000000,
    },
}


class APIProvider(LLMProvider):

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "", preset: str = ""):
        if preset and preset in PRESETS:
            p = PRESETS[preset]
            self.base_url = base_url or p["base_url"]
            self.model = model or p["model"]
            self._context_window = p.get("context_window", 128000)
        else:
            self.base_url = base_url or PRESETS["openai"]["base_url"]
            self.model = model or PRESETS["openai"]["model"]
            self._context_window = 128000

        self.api_key = api_key
        self._client = None

    def configure(self, api_key: str = "", base_url: str = "", model: str = "", preset: str = ""):
        if preset and preset in PRESETS:
            p = PRESETS[preset]
            self.base_url = base_url or p["base_url"]
            self.model = model or p["model"]
            self._context_window = p.get("context_window", 128000)
        else:
            if base_url:
                self.base_url = base_url
            if model:
                self.model = model

        if api_key:
            self.api_key = api_key

        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                logger.warning("openai package not installed")
                return None
        return self._client

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        client = self._get_client()
        return client is not None

    def generate(self, prompt: str, system_prompt: str = "", model: Optional[str] = None, temperature: float = 0.3, on_token=None) -> LLMResponse:
        client = self._get_client()
        if client is None:
            raise RuntimeError("API provider is not available")

        model_name = model or self.model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info("API generate: model=%s, base_url=%s, prompt_len=%d", model_name, self.base_url, len(prompt))

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )

        choice = response.choices[0]
        text = choice.message.content
        tokens = response.usage.total_tokens if response.usage else None

        return LLMResponse(
            text=text,
            model=model_name,
            provider="api",
            tokens_used=tokens
        )

    def list_models(self) -> List[str]:
        return [self.model]

    def provider_name(self) -> str:
        return "api"

    def context_window(self) -> int:
        return self._context_window
