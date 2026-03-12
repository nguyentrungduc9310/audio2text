import logging
import re
import time
from typing import Optional, List

from core.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3.5:4b"
MAX_PREDICT = 4096


class _StreamFilter:
    """Filter <think>...</think> blocks from streaming tokens."""

    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""
        self.suppressing = False
        self.checked_start = False

    def feed(self, text):
        if not self.callback:
            return
        self.buffer += text

        if not self.checked_start:
            stripped = self.buffer.lstrip()
            if stripped.startswith("<think>"):
                self.suppressing = True
                self.checked_start = True
            elif len(stripped) >= 7:  # len("<think>")
                self.checked_start = True
                self.callback(self.buffer)
                self.buffer = ""
                return
            else:
                return  # Wait for more data

        if self.suppressing:
            idx = self.buffer.find("</think>")
            if idx >= 0:
                self.suppressing = False
                after = self.buffer[idx + 8:]
                self.buffer = ""
                if after:
                    self.callback(after)
        else:
            self.callback(self.buffer)
            self.buffer = ""

    def flush(self):
        if self.buffer and self.callback and not self.suppressing:
            self.callback(self.buffer)
        self.buffer = ""


class OllamaProvider(LLMProvider):

    def __init__(self, default_model: str = DEFAULT_MODEL):
        self.default_model = default_model
        self._client = None
        self._health_client = None
        self._context_window_cache: dict = {}
        self._availability_cache: Optional[bool] = None
        self._availability_cache_time: float = 0
        self._models_cache: Optional[List[str]] = None
        self._models_cache_time: float = 0

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                import os
                host = os.environ.get('OLLAMA_HOST')
                self._client = ollama.Client(host=host, timeout=600.0) if host else ollama.Client(timeout=600.0)
            except ImportError:
                logger.warning("ollama package not installed")
                return None
        return self._client

    def _get_health_client(self):
        if self._health_client is None:
            try:
                import ollama
                import os
                host = os.environ.get('OLLAMA_HOST')
                self._health_client = ollama.Client(host=host, timeout=3.0) if host else ollama.Client(timeout=3.0)
            except ImportError:
                logger.warning("ollama package not installed")
                return None
        return self._health_client

    def is_available(self) -> bool:
        now = time.monotonic()
        if self._availability_cache is not None and (now - self._availability_cache_time) < 30:
            return self._availability_cache

        client = self._get_health_client()
        if client is None:
            self._availability_cache = False
            self._availability_cache_time = now
            return False
        try:
            client.list()
            self._availability_cache = True
            self._availability_cache_time = now
            return True
        except Exception as e:
            logger.warning("Ollama not available: %s", e)
            self._availability_cache = False
            self._availability_cache_time = now
            return False

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from text, including incomplete ones."""
        # Remove complete <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove incomplete <think> block at end (model hit token limit mid-thought)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
        return text.strip()

    def generate(self, prompt: str, system_prompt: str = "", model: Optional[str] = None, temperature: float = 0.3, on_token=None) -> LLMResponse:
        client = self._get_client()
        if client is None:
            raise RuntimeError("Ollama is not available")

        model_name = model or self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info("Ollama generate: model=%s, prompt_len=%d", model_name, len(prompt))

        chat_kwargs = dict(
            model=model_name,
            messages=messages,
            options={"temperature": temperature, "num_predict": MAX_PREDICT, "repeat_penalty": 1.2},
            stream=True,
            think=False,
        )

        # Use streaming to prevent TCP idle timeouts (Docker's host.docker.internal
        # proxy drops connections with no data transfer for ~2-3 minutes).
        # _strip_thinking() below handles any <think> tags in model output.
        stream = client.chat(**chat_kwargs)

        stream_filter = _StreamFilter(on_token)

        full_text = ""
        tokens = None
        for chunk in stream:
            if isinstance(chunk, dict):
                content = chunk.get("message", {}).get("content", "")
                if chunk.get("done"):
                    tokens = chunk.get("eval_count")
            else:
                msg = getattr(chunk, 'message', None)
                content = getattr(msg, 'content', '') if msg else ""
                if getattr(chunk, 'done', False):
                    tokens = getattr(chunk, 'eval_count', None)
            full_text += content
            if content:
                stream_filter.feed(content)

        stream_filter.flush()

        # Strip any <think> content that leaked through (older SDK or dict responses)
        full_text = self._strip_thinking(full_text)

        return LLMResponse(
            text=full_text,
            model=model_name,
            provider="ollama",
            tokens_used=tokens
        )

    def list_models(self) -> List[str]:
        now = time.monotonic()
        if self._models_cache is not None and (now - self._models_cache_time) < 60:
            return self._models_cache

        client = self._get_health_client()
        if client is None:
            self._models_cache = []
            self._models_cache_time = now
            return []
        try:
            response = client.list()
            # Handle both dict (old ollama) and typed object (new ollama) responses
            if isinstance(response, dict):
                models = response.get("models", [])
            else:
                models = getattr(response, 'models', []) or []
            result = []
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model", "")
                else:
                    name = getattr(m, 'model', '') or getattr(m, 'name', '')
                if name:
                    result.append(name)
            self._models_cache = result
            self._models_cache_time = now
            return result
        except Exception as e:
            logger.warning("Failed to list Ollama models: %s", e)
            self._models_cache = []
            self._models_cache_time = now
            return []

    def provider_name(self) -> str:
        return "ollama"

    def context_window(self) -> int:
        model_name = self.default_model
        if model_name in self._context_window_cache:
            return self._context_window_cache[model_name]

        ctx = 32768  # fallback
        client = self._get_health_client()
        if client is not None:
            try:
                info = client.show(model_name)
                if isinstance(info, dict):
                    params = info.get("model_info", {})
                else:
                    params = getattr(info, "model_info", {}) or {}
                # Keys vary: look for context_length in model_info values
                for key, val in (params.items() if isinstance(params, dict) else []):
                    if "context_length" in key:
                        ctx = int(val)
                        break
            except Exception as e:
                logger.debug("Could not query context window for %s: %s", model_name, e)

        self._context_window_cache[model_name] = ctx
        logger.info("Context window for %s: %d tokens", model_name, ctx)
        return ctx
