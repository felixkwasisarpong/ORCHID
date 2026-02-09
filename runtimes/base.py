"""Runtime client interfaces and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class RuntimeConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 20.0


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class ChatResult:
    content: str
    usage: TokenUsage
    cost_usd: float = 0.0


class RuntimeClient(Protocol):
    config: RuntimeConfig

    async def chat(self, messages: List[Dict[str, Any]], seed: int | None = None) -> ChatResult:
        ...
