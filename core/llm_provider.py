from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    tokens_used: Optional[int] = None


class LLMProvider(ABC):

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", model: Optional[str] = None, temperature: float = 0.3, on_token=None) -> LLMResponse:
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        pass

    @abstractmethod
    def provider_name(self) -> str:
        pass
